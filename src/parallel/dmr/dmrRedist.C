/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2026 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "dmrRedist.H"
#include "OSspecific.H"
#include "Pstream.H"
#include "fvMesh.H"
#include "fvMeshDistribute.H"
#include "decompositionMethod.H"
#include "polyDistributionMap.H"
#include "IOobjectList.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "pointFields.H"
#include "fvMeshTools.H"
#include "internalPolyPatch.H"
#include "processorPolyPatch.H"
#include "PstreamReduceOps.H"

extern "C"
{
    #include "dmr.h"
}

#include <mpi.h>
#include <glob.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <utility>
#include <vector>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// DMR-side subprocess and lifecycle logging. Set FOAM_DMR_LOG=0 to silence.
static bool dmrLogEnabled()
{
    static const bool enabled = []()
    {
        const char* env = std::getenv("FOAM_DMR_LOG");
        return !env || std::string(env) != "0";
    }();
    return enabled;
}


// log.dmr/ rather than logs/ because OpenFOAM's foamLog tool reserves
// the latter for residual data.
static std::string dmrLogPath
(
    const std::string& casePath,
    const std::string& logName
)
{
    if (!dmrLogEnabled()) return "/dev/null";
    return casePath + "/log.dmr/" + logName;
}


// UTC-timestamped lifecycle line appended to log.dmr/dmr_lifecycle.log
// and echoed to stderr. Stderr echo is unconditional so DMR's activity
// remains visible even when file logging is silenced.
static void dmrLifecycleLog
(
    const std::string& casePath,
    const std::string& msg
)
{
    std::fprintf(stderr, "DMR: %s\n", msg.c_str());

    if (!dmrLogEnabled()) return;

    Foam::mkDir(casePath + "/log.dmr");
    const std::string logPath = casePath + "/log.dmr/dmr_lifecycle.log";
    if (FILE* f = std::fopen(logPath.c_str(), "a"))
    {
        const auto now = std::time(nullptr);
        char ts[32];
        std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
        std::fprintf(f, "%s  %s\n", ts, msg.c_str());
        std::fclose(f);
    }
}


// Subprocess launcher. abortOnFailure=true (checkpoint/restart) calls
// MPI_Abort on non-zero exit so a stale case never reaches the new
// process group. abortOnFailure=false (finalize) returns the code.
static int dmrRunOrAbort
(
    const std::string& casePath,
    const std::string& logName,
    const std::string& utilityArgs,
    const bool abortOnFailure = true
)
{
    if (dmrLogEnabled())
    {
        Foam::mkDir(casePath + "/log.dmr");
    }

    const std::string logPath = dmrLogPath(casePath, logName);

    // Append so successive reconfigs accumulate in one file per utility.
    const std::string cmd =
        utilityArgs + " >> '" + logPath + "' 2>&1";

    const int ret = std::system(cmd.c_str());

    if (ret != 0 && abortOnFailure)
    {
        std::fprintf
        (
            stderr,
            "DMR: FATAL — '%s' exited with status %d; see '%s'.\n",
            utilityArgs.c_str(), ret, logPath.c_str()
        );
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return ret;
}


// Use foamDictionary as a subprocess rather than in-process Foam::dictionary
// I/O: the in-process round-trip silently no-ops on multi-region controlDicts
// that combine a regionSolvers sub-dict with a leading C++ comment block.
static void dmrUpdateDictEntry
(
    const std::string& casePath,
    const std::string& dictRelPath,
    const std::string& entry,
    const std::string& value
)
{
    const std::string cmd =
        "foamDictionary -entry " + entry + " -set '" + value + "'"
      + " '" + casePath + "/" + dictRelPath + "'";

    dmrRunOrAbort(casePath, "dmr_dict_updates.log", cmd);
}


// Return true if the given polyMesh directory contains a `faces` file
// (i.e. a full mesh topology snapshot, not just a points-only update).
static bool dmrPolyMeshHasTopology(const std::string& polyMeshDir)
{
    const std::string facesPath = polyMeshDir + "/faces";
    struct stat st;
    if (::stat(facesPath.c_str(), &st) == 0) return true;
    // OpenFOAM sometimes writes compressed: faces.gz
    const std::string facesGz = polyMeshDir + "/faces.gz";
    return ::stat(facesGz.c_str(), &st) == 0;
}


// Return true if the given time directory contains a complete polyMesh
// snapshot for THIS case (single-region or all-regions). A complete
// snapshot is one with a `faces` file (the universal topology marker).
// For all-regions cases, return true if any region under <time>/ has a
// complete polyMesh — decomposePar -allRegions will replay topology for
// every such region uniformly.
static bool dmrTimeDirHasMeshWrite
(
    const std::string& casePath,
    const std::string& timeName,
    bool allRegions
)
{
    if (!allRegions)
    {
        return dmrPolyMeshHasTopology
        (
            casePath + "/" + timeName + "/polyMesh"
        );
    }

    // all-regions: glob <case>/<time>/*/polyMesh
    const std::string pattern =
        casePath + "/" + timeName + "/*/polyMesh";
    glob_t g;
    if (glob(pattern.c_str(), 0, nullptr, &g) != 0)
    {
        return false;
    }
    bool any = false;
    for (size_t i = 0; i < g.gl_pathc && !any; ++i)
    {
        if (dmrPolyMeshHasTopology(g.gl_pathv[i])) any = true;
    }
    globfree(&g);
    return any;
}


// Build the comma-separated -time argument for decomposePar:
// "<latest_mesh_write>,<restartTime>" for moving-mesh cases,
// "<restartTime>" otherwise. Lets decomposePar evolve the mesh forward
// from the most recent topology snapshot — see bug 0004368.
static std::string dmrBuildDecomposeTimeArg
(
    const std::string& casePath,
    const std::string& restartTime,
    bool allRegions
)
{
    const std::string pattern = casePath + "/[0-9]*";
    glob_t g;
    if (glob(pattern.c_str(), 0, nullptr, &g) != 0)
    {
        return restartTime;
    }

    const double restartT = std::stod(restartTime);

    double bestT = -1.0;
    std::string bestName;
    for (size_t i = 0; i < g.gl_pathc; ++i)
    {
        std::string p(g.gl_pathv[i]);
        const std::string name = p.substr(p.find_last_of('/') + 1);
        double t;
        try { t = std::stod(name); }
        catch (const std::exception&) { continue; }

        if (t >= restartT) continue;
        if (!dmrTimeDirHasMeshWrite(casePath, name, allRegions)) continue;
        if (t > bestT)
        {
            bestT = t;
            bestName = name;
        }
    }
    globfree(&g);

    if (bestName.empty()) return restartTime;
    return bestName + "," + restartTime;
}


// Largest float-parsed numeric time directory name. Empty string if none.
// The "[0-9]*" pattern excludes constant, system, processor*, log.dmr.
static std::string dmrLatestTimeName(const std::string& casePath)
{
    const std::string pattern = casePath + "/[0-9]*";
    glob_t g;
    if (glob(pattern.c_str(), 0, nullptr, &g) != 0)
    {
        return "";
    }

    double maxT = -1.0;
    std::string maxName;
    for (size_t i = 0; i < g.gl_pathc; ++i)
    {
        std::string p(g.gl_pathv[i]);
        const auto slash = p.find_last_of('/');
        const std::string name = p.substr(slash + 1);
        try
        {
            const double t = std::stod(name);
            if (t > maxT)
            {
                maxT = t;
                maxName = name;
            }
        }
        catch (const std::exception&)
        {
            // Not a numeric dir — skip.
        }
    }
    globfree(&g);
    return maxName;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Reconfiguration direction reported by DMR analytics.
enum class DmrDir { Unknown = 0, Shrink = 1, Grow = 2 };

struct DmrWorldInfo
{
    int nOld = 0;
    int nNew = 0;
    DmrDir dir = DmrDir::Unknown;
};


// Rank 0 queries DMR analytics + procs_next_* getters; result is broadcast
// to every rank. Valid window: between dmr_reconfigure() returning
// DMR_REDIST_FINALIZE and dmr_finalize() running — i.e. inside the redist
// callback in DMR_AUTO.
static bool dmrQueryWorldInfo(DmrWorldInfo& w)
{
    MPI_Comm_size(MPI_COMM_WORLD, &w.nOld);

    int dir = static_cast<int>(DmrDir::Unknown);
    int delta = 0;

    int myRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (myRank == 0)
    {
        DMRAnalytics a{};
        if (dmr_get_analytics(&a) == DMR_SUCCESS)
        {
            switch (a.event)
            {
                case DMR_EVENT_START_EXPAND_SLURM:
                case DMR_EVENT_START_EXPAND_MPI:
                    dir = static_cast<int>(DmrDir::Grow);
                    delta = dmr_get_procs_next_expand();
                    break;
                case DMR_EVENT_START_SHRINK:
                    dir = static_cast<int>(DmrDir::Shrink);
                    delta = dmr_get_procs_next_shrink();
                    break;
                default:
                    break;
            }
        }
    }

    int buf[2] = {dir, delta};
    MPI_Bcast(buf, 2, MPI_INT, 0, MPI_COMM_WORLD);
    dir = buf[0];
    delta = buf[1];

    w.dir = static_cast<DmrDir>(dir);
    if (w.dir == DmrDir::Grow)
    {
        w.nNew = w.nOld + delta;
    }
    else if (w.dir == DmrDir::Shrink)
    {
        w.nNew = w.nOld - delta;
    }
    else
    {
        w.nNew = w.nOld;
    }

    return w.dir != DmrDir::Unknown && delta > 0 && w.nNew > 0;
}


static const char* dmrDirToStr(DmrDir d)
{
    switch (d)
    {
        case DmrDir::Shrink: return "SHRINK";
        case DmrDir::Grow:   return "GROW";
        default:             return "UNKNOWN";
    }
}


// Plain-text key=value per line so the NEW group can read it without
// depending on OF's dictionary machinery (which is not initialised yet
// at dmrRestart time).
static void dmrWriteWorldMeta
(
    const std::string& casePath,
    const DmrWorldInfo& w
)
{
    Foam::mkDir(casePath + "/log.dmr");
    const std::string path = casePath + "/log.dmr/.world.meta";

    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;

    std::fprintf(f, "n_old=%d\n", w.nOld);
    std::fprintf(f, "n_new=%d\n", w.nNew);
    std::fprintf(f, "direction=%s\n", dmrDirToStr(w.dir));
    std::fclose(f);
    ::sync();
}


// Rank 0 reads, broadcasts. Returns true only on a fully-populated meta;
// stale or partial files are treated as missing so the caller knows it
// can't rely on the direction.
static bool dmrReadWorldMeta
(
    const std::string& casePath,
    DmrWorldInfo& w
)
{
    int myRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    int payload[3] = {0, 0, static_cast<int>(DmrDir::Unknown)};
    int found = 0;

    if (myRank == 0)
    {
        const std::string path = casePath + "/log.dmr/.world.meta";
        FILE* f = std::fopen(path.c_str(), "r");
        if (f)
        {
            char line[128];
            int n_old = 0, n_new = 0;
            char dirStr[32] = "UNKNOWN";
            while (std::fgets(line, sizeof(line), f))
            {
                if (std::sscanf(line, "n_old=%d", &n_old) == 1) continue;
                if (std::sscanf(line, "n_new=%d", &n_new) == 1) continue;
                std::sscanf(line, "direction=%31s", dirStr);
            }
            std::fclose(f);

            DmrDir d = DmrDir::Unknown;
            if      (std::string(dirStr) == "SHRINK") d = DmrDir::Shrink;
            else if (std::string(dirStr) == "GROW")   d = DmrDir::Grow;

            payload[0] = n_old;
            payload[1] = n_new;
            payload[2] = static_cast<int>(d);
            found = (n_old > 0 && n_new > 0 && d != DmrDir::Unknown) ? 1 : 0;
        }
    }

    MPI_Bcast(&found, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!found) return false;

    MPI_Bcast(payload, 3, MPI_INT, 0, MPI_COMM_WORLD);
    w.nOld = payload[0];
    w.nNew = payload[1];
    w.dir = static_cast<DmrDir>(payload[2]);
    return true;
}


// Load all on-disk fields of a given GeoField type into a PtrList,
// registering each with the supplied mesh. Matches the loading pattern
// redistributePar uses on its freshly-constructed mesh.
template<class GeoField>
static void dmrLoadFields
(
    const typename GeoField::Mesh& mesh,
    IOobjectList& allObjects,
    PtrList<GeoField>& fields
)
{
    IOobjectList objects(allObjects.lookupClass(GeoField::typeName));
    const wordList names(objects.toc());

    fields.setSize(names.size());
    forAll(names, i)
    {
        IOobject& io = objects[names[i]];
        io.writeOpt() = IOobject::AUTO_WRITE;
        fields.set(i, new GeoField(io, mesh));
    }
}


// Construct a fresh fvMesh from disk (non-registered, so it doesn't clash
// with the live mesh still in runTime's registry), load every field type
// into it, run fvMeshDistribute, and write the new layout. Mirrors the
// body of redistributePar.C's main() — calling fvMeshDistribute against
// this clean mesh avoids the receiver-side deadlock that happens when it
// is called against a mesh carrying live solver state. The region argument
// is the polyMesh name (polyMesh::defaultRegion for single-region cases,
// or one of the case's named regions for foamMultiRun).
static void dmrRunFreshRedistribute
(
    Time& runTime,
    const std::string& timeName,
    const word& regionName
)
{
    fileName meshSubDir;
    if (regionName == polyMesh::defaultRegion)
    {
        meshSubDir = polyMesh::meshSubDir;
    }
    else
    {
        meshSubDir = regionName / polyMesh::meshSubDir;
    }

    const fileName masterInstDir =
        runTime.findInstance(meshSubDir, "points");

    fvMesh mesh
    (
        IOobject
        (
            regionName,
            masterInstDir,
            runTime,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE,
            false
        )
    );

    autoPtr<decompositionMethod> distributor
    (
        decompositionMethod::NewDistributor
        (
            decompositionMethod::decomposeParDict(runTime)
        )
    );
    const labelList finalDecomp =
        distributor().decompose(mesh, mesh.cellCentres());

    IOobjectList allObjects(mesh, runTime.name());

    PtrList<volScalarField>              volScalars;
    PtrList<volVectorField>              volVectors;
    PtrList<volSphericalTensorField>     volSphereTensors;
    PtrList<volSymmTensorField>          volSymmTensors;
    PtrList<volTensorField>              volTensors;
    PtrList<surfaceScalarField>          surfScalars;
    PtrList<surfaceVectorField>          surfVectors;
    PtrList<surfaceSphericalTensorField> surfSphereTensors;
    PtrList<surfaceSymmTensorField>      surfSymmTensors;
    PtrList<surfaceTensorField>          surfTensors;

    dmrLoadFields(mesh, allObjects, volScalars);
    dmrLoadFields(mesh, allObjects, volVectors);
    dmrLoadFields(mesh, allObjects, volSphereTensors);
    dmrLoadFields(mesh, allObjects, volSymmTensors);
    dmrLoadFields(mesh, allObjects, volTensors);
    dmrLoadFields(mesh, allObjects, surfScalars);
    dmrLoadFields(mesh, allObjects, surfVectors);
    dmrLoadFields(mesh, allObjects, surfSphereTensors);
    dmrLoadFields(mesh, allObjects, surfSymmTensors);
    dmrLoadFields(mesh, allObjects, surfTensors);

    // Point fields (e.g. pointDisplacement on moving meshes). Their mesh
    // type is pointMesh, not fvMesh, so they load against pointMesh::New.
    // fvMeshDistribute maps every field registered on the mesh, so these
    // must be loaded (and thereby registered) before distribute() runs.
    pointMesh& pMesh = const_cast<pointMesh&>(pointMesh::New(mesh));

    PtrList<pointScalarField>          pointScalars;
    PtrList<pointVectorField>          pointVectors;
    PtrList<pointSphericalTensorField> pointSphereTensors;
    PtrList<pointSymmTensorField>      pointSymmTensors;
    PtrList<pointTensorField>          pointTensors;

    dmrLoadFields(pMesh, allObjects, pointScalars);
    dmrLoadFields(pMesh, allObjects, pointVectors);
    dmrLoadFields(pMesh, allObjects, pointSphereTensors);
    dmrLoadFields(pMesh, allObjects, pointSymmTensors);
    dmrLoadFields(pMesh, allObjects, pointTensors);

    // fvMeshDistribute requires an internal-type patch when fields are
    // registered (findInternalPatch). The on-disk case must not carry
    // one (it clashes with the non-conformal solver machinery), so add
    // a transient zero-size internalPolyPatch to this private mesh copy
    // after field loading (fvMesh::addPatch extends every registered
    // field's boundary) and remove it again before writing. Same
    // pattern as createNonConformalCouples' patch insertion.
    const word dmrPatchName("dmrTransientInternal");
    bool dmrAddedPatch = false;
    {
        const polyBoundaryMesh& pbm = mesh.poly().boundary();

        bool hasInternal = false;
        forAll(pbm, patchi)
        {
            if (isA<internalPolyPatch>(pbm[patchi]))
            {
                hasInternal = true;
                break;
            }
        }

        if (!hasInternal)
        {
            // Zero-size, positioned where fvMeshTools::addPatch inserts
            // it: before the first processor patch, or at the end.
            label start = mesh.nFaces();
            forAll(pbm, patchi)
            {
                if (isA<processorPolyPatch>(pbm[patchi]))
                {
                    start = pbm[patchi].start();
                    break;
                }
            }

            fvMeshTools::addPatch
            (
                mesh,
                internalPolyPatch
                (
                    dmrPatchName,
                    0,
                    start,
                    pbm.size(),
                    pbm
                )
            );
            dmrAddedPatch = true;
        }
    }

    fvMeshDistribute(mesh).distribute(finalDecomp);

    // Remove the transient patch before writing. It must be empty
    // again (exposed faces are re-homed onto processor patches by the
    // end of distribute); the check is reduced so all ranks take the
    // same branch, since reorderPatches is collective.
    if (dmrAddedPatch)
    {
        const polyBoundaryMesh& pbm = mesh.poly().boundary();
        const label patchi = pbm.findIndex(dmrPatchName);

        label globalSize = patchi == -1 ? 0 : pbm[patchi].size();
        reduce(globalSize, sumOp<label>());

        if (patchi != -1 && globalSize == 0)
        {
            labelList oldToNew(pbm.size());
            label newi = 0;
            forAll(oldToNew, i)
            {
                if (i != patchi)
                {
                    oldToNew[i] = newi++;
                }
            }
            oldToNew[patchi] = newi;

            fvMeshTools::reorderPatches(mesh, oldToNew, newi, true);
        }
        else if (patchi != -1)
        {
            WarningInFunction
                << "Transient internal patch holds " << globalSize
                << " faces after distribute; keeping it in the written"
                << " mesh." << endl;
        }
    }

    mesh.setInstance(timeName);
    mesh.write();
}


// SHRINK on the OLD group:
// 1. Flush the live solver state to processor*/timeName/ (writeNow).
// 2. Update decomposeParDict for the new world size + scotch.
// 3. Run the redistribute on a fresh mesh loaded from that just-written
//    state (see dmrRunFreshRedistribute for the rationale).
// 4. Sweep processor{N_new..N_old-1}/ - those are empty after the
//    redistribute and the NEW group will only look at 0..N_new-1.
static void dmrShrinkInPlace
(
    Time& runTime,
    const DmrWorldInfo& world,
    bool isDmrForced,
    bool /*allRegions*/
)
{
    const std::string casePath = runTime.globalPath();
    const std::string timeName = runTime.name();

    runTime.writeNow();
    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        dmrLifecycleLog
        (
            casePath,
            "Checkpoint at t=" + timeName
          + (isDmrForced ? " (DMR-forced)" : " (writeInterval)")
        );
        dmrWriteWorldMeta(casePath, world);
        dmrLifecycleLog
        (
            casePath,
            "Reconfig SHRINK: N_old=" + std::to_string(world.nOld)
          + " -> N_new=" + std::to_string(world.nNew)
        );

        dmrUpdateDictEntry
        (
            casePath, "system/decomposeParDict",
            "numberOfSubdomains", std::to_string(world.nNew)
        );
        dmrUpdateDictEntry
        (
            casePath, "system/decomposeParDict",
            "decomposer", "scotch"
        );
        dmrUpdateDictEntry
        (
            casePath, "system/decomposeParDict",
            "distributor", "scotch"
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Snapshot the live region names BEFORE the redistribute loop. Each
    // call to dmrRunFreshRedistribute constructs a non-registered fvMesh
    // that does not appear in the registry, so iterating the live mesh
    // table once up front avoids any iterator-vs-mutation hazard.
    wordList regionNames;
    {
        HashTable<fvMesh*> liveRegions(runTime.lookupClass<fvMesh>());
        regionNames.setSize(liveRegions.size());
        label i = 0;
        forAllIter(HashTable<fvMesh*>, liveRegions, iter)
        {
            regionNames[i++] = iter()->name();
        }
    }

    forAll(regionNames, i)
    {
        dmrRunFreshRedistribute(runTime, timeName, regionNames[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        for (int p = world.nNew; p < world.nOld; ++p)
        {
            Foam::rmDir(casePath + "/processor" + std::to_string(p));
        }
        ::sync();

        dmrLifecycleLog
        (
            casePath,
            "Shrink redistribute complete ("
          + std::to_string(regionNames.size()) + " region(s))."
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void dmrCheckpoint(Time& runTime, bool allRegions)
{
    DmrWorldInfo world;
    const bool worldOk = dmrQueryWorldInfo(world);
    const bool isDmrForced = !runTime.writeTime();

    if (worldOk && world.dir == DmrDir::Shrink)
    {
        dmrShrinkInPlace(runTime, world, isDmrForced, allRegions);
        return;
    }

    // GROW / Unknown: legacy reconstructPar path. Replaced by an
    // in-process redistribute on the NEW group in Phase 3.
    runTime.writeNow();

    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        const std::string casePath = runTime.globalPath();
        const std::string timeName = runTime.name();

        dmrLifecycleLog
        (
            casePath,
            "Checkpoint at t=" + timeName
          + (isDmrForced ? " (DMR-forced)" : " (writeInterval)")
        );

        if (worldOk)
        {
            dmrWriteWorldMeta(casePath, world);
            dmrLifecycleLog
            (
                casePath,
                std::string("Reconfig ") + dmrDirToStr(world.dir)
              + ": N_old=" + std::to_string(world.nOld)
              + " -> N_new=" + std::to_string(world.nNew)
            );
        }
        else
        {
            dmrLifecycleLog
            (
                casePath,
                "Reconfig direction unknown (DMR analytics unavailable)."
            );
        }

        dmrLifecycleLog(casePath, "Reconstructing processor directories...");

        // -newTimes merges only times not yet in serial. -rm is omitted
        // because on dynamic meshes removing processor*/<earlier>/
        // mid-walk trips the facesInstance check in
        // domainDecomposition::readUpdateReconstruct; we sweep below
        // after reconstructPar has exited.
        std::string cmd =
            "reconstructPar -newTimes -case '" + casePath + "'";
        if (allRegions) cmd += " -allRegions";

        dmrRunOrAbort(casePath, "dmr_reconstruct.log", cmd);

        // Remove processor*/<time>/ only when the serial <time>/ exists,
        // so a silent reconstruct no-op cannot delete data.
        {
            glob_t pg;
            const std::string ppattern = casePath + "/processor*";
            if (glob(ppattern.c_str(), GLOB_ONLYDIR, nullptr, &pg) == 0)
            {
                for (size_t i = 0; i < pg.gl_pathc; ++i)
                {
                    const std::string procDir(pg.gl_pathv[i]);
                    glob_t tg;
                    const std::string tpattern = procDir + "/[0-9]*";
                    if (glob(tpattern.c_str(), GLOB_ONLYDIR, nullptr, &tg) == 0)
                    {
                        for (size_t j = 0; j < tg.gl_pathc; ++j)
                        {
                            const std::string procTimeDir(tg.gl_pathv[j]);
                            const auto slash = procTimeDir.find_last_of('/');
                            const std::string tName =
                                procTimeDir.substr(slash + 1);
                            const std::string serialTimeDir =
                                casePath + "/" + tName;
                            if (Foam::isDir(serialTimeDir))
                            {
                                Foam::rmDir(procTimeDir);
                            }
                        }
                        globfree(&tg);
                    }
                }
                globfree(&pg);
            }
        }

        // sync() guards against filesystem coherence delays we observed
        // on Docker bind mounts before the new group's decomposePar
        // reads what we wrote. No-op on POSIX-coherent FS (GPFS, Lustre,
        // ext4).
        ::sync();

        dmrLifecycleLog(casePath, "Reconstruction complete.");

        // Sentinel-based checkpoint cleanup. The sweep deletes prior
        // off-cadence DMR-forced checkpoints — their data is already
        // in memory and is not needed for restart — EXCEPT any time
        // directory whose polyMesh/ holds a mesh-write (faces file):
        // those are anchors that a future decomposePar's -time chain
        // needs (bug 0004368).
        //
        // OF time names compare as floats, not as version-sort strings
        // (which would mis-order "0.17" before "0.085").
        if (isDmrForced)
        {
            const std::string sentinel =
                casePath + "/" + timeName + "/.dmr_checkpoint";
            if (FILE* s = std::fopen(sentinel.c_str(), "w"))
                std::fclose(s);

            const std::string pattern = casePath + "/*/.dmr_checkpoint";
            glob_t g;
            if (glob(pattern.c_str(), 0, nullptr, &g) == 0)
            {
                const std::string suffix = "/.dmr_checkpoint";
                std::vector<std::pair<double, std::string>> dirs;
                dirs.reserve(g.gl_pathc);
                for (size_t i = 0; i < g.gl_pathc; ++i)
                {
                    std::string p(g.gl_pathv[i]);
                    p.resize(p.size() - suffix.size());
                    const auto slash = p.find_last_of('/');
                    const std::string tName = p.substr(slash + 1);
                    try
                    {
                        dirs.emplace_back(std::stod(tName), p);
                    }
                    catch (const std::exception&) {}
                }
                globfree(&g);

                std::sort(dirs.begin(), dirs.end());

                // Drop newest (last); remove rest unless mesh-write anchor.
                for (size_t i = 0; i + 1 < dirs.size(); ++i)
                {
                    const std::string& abs = dirs[i].second;
                    const auto slash = abs.find_last_of('/');
                    const std::string tName = abs.substr(slash + 1);

                    if (dmrTimeDirHasMeshWrite(casePath, tName, allRegions))
                    {
                        // Keep mesh-write anchors; just drop the sentinel
                        // so the next sweep does not retry.
                        const std::string thisSentinel =
                            abs + "/.dmr_checkpoint";
                        ::unlink(thisSentinel.c_str());
                        continue;
                    }

                    Foam::rmDir(abs);
                }
            }
        }
    }

    // Hold old-group ranks in lockstep before they exit DMR_AUTO and
    // proceed into MPI_Finalize via dmr_reconfigure().
    MPI_Barrier(MPI_COMM_WORLD);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void dmrRestart(const std::string& casePath, bool allRegions)
{
    // Read direction + N_old written by the OLD group at checkpoint.
    // Collective; broadcasts to every rank.
    DmrWorldInfo world;
    const bool worldOk = dmrReadWorldMeta(casePath, world);

    // Subprocesses spawned here run in isolated MPI environments, so
    // there is no interaction with DMR_INTERCOMM.
    int newSize, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &newSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    if (myRank == 0)
    {
        dmrLifecycleLog
        (
            casePath,
            "Restarting with " + std::to_string(newSize) + " processes."
        );

        if (worldOk)
        {
            dmrLifecycleLog
            (
                casePath,
                std::string("Restart ") + dmrDirToStr(world.dir)
              + ": N_old=" + std::to_string(world.nOld)
              + ", N_new(meta)=" + std::to_string(world.nNew)
              + ", MPI_Comm_size=" + std::to_string(newSize)
            );
            if (world.nNew != newSize)
            {
                dmrLifecycleLog
                (
                    casePath,
                    "WARNING: .world.meta n_new ("
                  + std::to_string(world.nNew)
                  + ") disagrees with MPI_Comm_size ("
                  + std::to_string(newSize) + ")."
                );
            }
        }
        else
        {
            dmrLifecycleLog
            (
                casePath,
                "Restart: no .world.meta (first run or stale)."
            );
        }

        dmrUpdateDictEntry
        (
            casePath, "system/decomposeParDict",
            "numberOfSubdomains", std::to_string(newSize)
        );

        // Force scotch: geometric decomposers (hierarchical, simple)
        // require n-coefficients that no longer hold once DMR has
        // changed the process count.
        dmrUpdateDictEntry
        (
            casePath, "system/decomposeParDict",
            "decomposer", "scotch"
        );

        dmrUpdateDictEntry
        (
            casePath, "system/controlDict",
            "startFrom", "latestTime"
        );

        // TODO multi-region: per-region system/<region>/decomposeParDict
        // and controlDict overrides are not handled. The root dicts
        // suffice for the -allRegions cases tested so far.

        if (worldOk && world.dir == DmrDir::Shrink)
        {
            // OLD group has already written processor0..N_new-1/ via
            // fvMeshDistribute. decomposePar would overwrite that from
            // the stale serial dirs.
            dmrLifecycleLog
            (
                casePath,
                "Shrink restart: data already in N_new processor layout."
            );
        }
        else
        {
            const std::string restartTime = dmrLatestTimeName(casePath);
            if (restartTime.empty())
            {
                std::fprintf
                (
                    stderr,
                    "DMR: FATAL — no time directories found in '%s'.\n",
                    casePath.c_str()
                );
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // For moving-mesh cases the polyMesh topology is only written
            // at mesh-swap times; between those, time directories contain
            // only a `points` file. decomposePar evolves topology forward
            // via a comma-separated time chain (e.g. -time 0.1,0.8) — see
            // bug 0004368 on bugs.openfoam.org.
            const std::string timeArg =
                dmrBuildDecomposeTimeArg(casePath, restartTime, allRegions);

            std::string cmd =
                "decomposePar -force -time " + timeArg + " -cellProc"
              + " -case '" + casePath + "'";
            if (allRegions) cmd += " -allRegions";

            dmrRunOrAbort(casePath, "dmr_decompose.log", cmd);

            // Engine-style cases keep additional sub-meshes under
            // constant/meshes/<name> that -allRegions does not visit.
            const std::string subMeshGlob =
                casePath + "/constant/meshes/*";
            glob_t mg;
            if (glob(subMeshGlob.c_str(), GLOB_ONLYDIR, nullptr, &mg) == 0)
            {
                for (size_t i = 0; i < mg.gl_pathc; ++i)
                {
                    const std::string p(mg.gl_pathv[i]);
                    const auto slash = p.find_last_of('/');
                    const std::string meshName = p.substr(slash + 1);
                    const std::string mcmd =
                        "decomposePar -mesh " + meshName
                      + " -region fluid -case '" + casePath + "'";
                    dmrRunOrAbort(casePath, "dmr_decompose.log", mcmd);
                }
                globfree(&mg);
            }

            dmrLifecycleLog(casePath, "Decomposition complete.");
        }
    }

    // Synchronise before setRootCase.H re-initialises OpenFOAM in
    // parallel mode on the new world.
    MPI_Barrier(MPI_COMM_WORLD);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void dmrFinalize(const std::string& casePath, bool allRegions)
{
    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        dmrLifecycleLog(casePath, "Final reconstruction to serial...");

        // -latestTime (not -newTimes): with in-process SHRINK each
        // reconfiguration writes a new processor-side polyMesh at the SHRINK
        // time, which trips reconstructPar's mesh-evolution check (bug
        // 0004368-style) when it tries to chain through the whole time
        // sequence. Reconstructing the final time is enough for post-run
        // analysis; intermediate serial snapshots can be produced manually
        // with `reconstructPar -time T` if needed.
        std::string cmd =
            "reconstructPar -latestTime -rm -case '" + casePath + "'";
        if (allRegions) cmd += " -allRegions";

        // Soft fail: the simulation has already finished, so on
        // reconstruct failure leave processor* dirs in place for
        // inspection rather than killing the job.
        const int ret =
            dmrRunOrAbort(casePath, "dmr_final.log", cmd, /*abort=*/false);

        if (ret != 0)
        {
            WarningInFunction
                << "DMR: final reconstructPar failed (exit " << ret
                << "). Processor directories preserved for inspection; "
                << "see " << dmrLogPath(casePath, "dmr_final.log").c_str()
                << nl;
        }
        else
        {
            dmrLifecycleLog(casePath, "Final reconstruction complete.");

            // Cleanup processor* dirs only on successful reconstruct;
            // on failure leave them in place for the user to inspect.
            glob_t pg;
            const std::string ppattern = casePath + "/processor*";
            if (glob(ppattern.c_str(), GLOB_ONLYDIR, nullptr, &pg) == 0)
            {
                for (size_t i = 0; i < pg.gl_pathc; ++i)
                {
                    Foam::rmDir(pg.gl_pathv[i]);
                }
                globfree(&pg);
            }
        }

        // Sweep any remaining DMR-forced sentinel dirs — superseded by
        // the final reconstructPar above.
        const std::string pattern = casePath + "/*/.dmr_checkpoint";
        glob_t g;
        if (glob(pattern.c_str(), 0, nullptr, &g) == 0)
        {
            const std::string suffix = "/.dmr_checkpoint";
            for (size_t i = 0; i < g.gl_pathc; ++i)
            {
                std::string p(g.gl_pathv[i]);
                p.resize(p.size() - suffix.size());
                Foam::rmDir(p);
            }
            globfree(&g);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
