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
#include "fvMeshSubset.H"
#include "loadOrCreateMesh.H"
#include "HashSet.H"
#include "pointIOField.H"
#include "OStringStream.H"
#include "IStringStream.H"

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


// Largest numeric time in processor0/. In the all-parallel pipeline
// checkpoints exist only in the processor layout (no serial time
// dirs), so this — not dmrLatestTimeName — names the restart time.
static std::string dmrLatestProcTimeName(const std::string& casePath)
{
    return dmrLatestTimeName(casePath + "/processor0");
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


// Grow-side field loader: ranks without a mesh have no field files, so
// the master interpolates each field to zero size (via the subsetter)
// and sends it to them, keeping the field set identical on every rank
// before fvMeshDistribute runs. Mirrors redistributePar.C's readFields.
template<class GeoField>
static void dmrLoadFieldsGrow
(
    const boolList& haveMesh,
    const typename GeoField::Mesh& mesh,
    const autoPtr<fvMeshSubset>& subsetterPtr,
    IOobjectList& allObjects,
    PtrList<GeoField>& fields
)
{
    IOobjectList objects(allObjects.lookupClass(GeoField::typeName));

    wordList masterNames(objects.toc());
    Pstream::scatter(masterNames);

    if (haveMesh[Pstream::myProcNo()] && objects.toc() != masterNames)
    {
        FatalErrorInFunction
            << "differing fields of type " << GeoField::typeName
            << " on processors." << endl
            << "Master has:" << masterNames << endl
            << Pstream::myProcNo() << " has:" << objects.toc()
            << abort(FatalError);
    }

    fields.setSize(masterNames.size());

    if (Pstream::master())
    {
        forAll(masterNames, i)
        {
            IOobject& io = objects[masterNames[i]];
            io.writeOpt() = IOobject::AUTO_WRITE;

            fields.set(i, new GeoField(io, mesh));

            // Zero-sized copy to every rank without a mesh
            if (subsetterPtr.valid())
            {
                tmp<GeoField> tsubfld = subsetterPtr().interpolate(fields[i]);

                for (label proci = 1; proci < Pstream::nProcs(); proci++)
                {
                    if (!haveMesh[proci])
                    {
                        OPstream toProc(Pstream::commsTypes::blocking, proci);
                        toProc<< tsubfld();
                    }
                }
            }
        }
    }
    else if (!haveMesh[Pstream::myProcNo()])
    {
        forAll(masterNames, i)
        {
            IPstream fromMaster
            (
                Pstream::commsTypes::blocking,
                Pstream::masterNo()
            );
            dictionary fieldDict(fromMaster);

            fields.set
            (
                i,
                new GeoField
                (
                    IOobject
                    (
                        masterNames[i],
                        mesh.db().time().name(),
                        mesh.db(),
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE
                    ),
                    mesh,
                    fieldDict
                )
            );
        }
    }
    else
    {
        forAll(masterNames, i)
        {
            IOobject& io = objects[masterNames[i]];
            io.writeOpt() = IOobject::AUTO_WRITE;

            fields.set(i, new GeoField(io, mesh));
        }
    }
}


// Reference-configuration transport. points0MotionSolver and its
// family (solidBodyMeshMotion, the displacement solvers, the
// interpolator mover) rebuild their reference points at startup via
// readPoints0: a points0 field at a case-root time instance if one
// exists, else constant/<region>/polyMesh/points. Neither survives a
// redistribution by itself: the constant mesh stays in the source
// layout, and does not exist at all on ranks added by a grow, so the
// restarted solver would read a reference of the wrong size
// (points0MotionSolver's size check FATALs). Resolve the reference on
// the source layout the same way readPoints0 does (checkpoint-carried
// field, else newest time-instance points0 at this region, else
// per-rank constant points, valid precisely when readPoints0's own
// fallback is) and register it on the mesh so fvMeshDistribute
// transports it with the other point fields. dmrWritePoints0Constant
// then re-creates the constant points file in the new layout: the
// artefact a serial decomposePar would have produced, and the only
// channel readPoints0's region-blind time search can see for
// non-default regions.
static void dmrEnsurePoints0
(
    Time& runTime,
    fvMesh& mesh,
    pointMesh& pMesh,
    const boolList& haveMesh,                   // empty: all ranks have data
    const autoPtr<fvMeshSubset>& subsetterPtr,  // unset: all ranks have data
    const word& regionName,
    PtrList<pointVectorField>& pointVectors
)
{
    forAll(pointVectors, i)
    {
        if (pointVectors[i].name() == "points0")
        {
            // The checkpoint carries it; already loaded and transported
            return;
        }
    }

    const fileName regionDir =
        regionName == polyMesh::defaultRegion
      ? fileName(".")
      : fileName(regionName);

    // Only motion cases carry a reference configuration. The master
    // decides. The dictionary lives at CASE level only: decomposePar
    // copies meshes and fields into processor trees, never system or
    // constant dictionaries, so probing runTime.path() (the processor
    // directory) always misses (engine, 2026-07-13)
    bool moving = false;
    word inst;
    if (Pstream::master())
    {
        moving = isFile
        (
            fileName(runTime.globalPath())
           /runTime.constant()/regionDir/"dynamicMeshDict"
        );

        if (moving)
        {
            const instantList times = runTime.times();
            forAllReverse(times, i)
            {
                if (times[i].name() == runTime.constant()) continue;

                if (isFile(runTime.path()/times[i].name()/regionDir/"points0"))
                {
                    inst = times[i].name();
                    break;
                }
            }
        }
    }
    Pstream::scatter(moving);
    Pstream::scatter(inst);

    if (!moving)
    {
        return;
    }

    const bool have =
        haveMesh.empty() || haveMesh[Pstream::myProcNo()];

    autoPtr<pointVectorField> p0;

    if (inst != word::null)
    {
        // A previous write (a topo-change or an earlier reconfiguration)
        // left a points0 field in the current source layout: load it,
        // AUTO_WRITE so the rewritten copy keeps readPoints0's
        // time-instance branch working after the reconfiguration
        if (Pstream::master())
        {
            p0.reset
            (
                new pointVectorField
                (
                    IOobject
                    (
                        "points0",
                        inst,
                        mesh,
                        IOobject::MUST_READ,
                        IOobject::AUTO_WRITE,
                        true
                    ),
                    pMesh
                )
            );

            if (subsetterPtr.valid())
            {
                tmp<pointVectorField> tsubfld =
                    subsetterPtr().interpolate(p0());

                for (label proci = 1; proci < Pstream::nProcs(); proci++)
                {
                    if (!haveMesh[proci])
                    {
                        OPstream toProc(Pstream::commsTypes::blocking, proci);
                        toProc<< tsubfld();
                    }
                }
            }
        }
        else if (!have)
        {
            IPstream fromMaster
            (
                Pstream::commsTypes::blocking,
                Pstream::masterNo()
            );
            dictionary fieldDict(fromMaster);

            p0.reset
            (
                new pointVectorField
                (
                    IOobject
                    (
                        "points0",
                        runTime.name(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE,
                        true
                    ),
                    pMesh,
                    fieldDict
                )
            );
        }
        else
        {
            p0.reset
            (
                new pointVectorField
                (
                    IOobject
                    (
                        "points0",
                        inst,
                        mesh,
                        IOobject::MUST_READ,
                        IOobject::AUTO_WRITE,
                        true
                    ),
                    pMesh
                )
            );
        }
    }
    else
    {
        // First reconfiguration of the case: synthesise from the
        // per-rank constant points (readPoints0's own fallback), which
        // every data-carrying rank still has. NO_WRITE: transported in
        // memory, persisted only through the constant dump
        p0.reset
        (
            new pointVectorField
            (
                IOobject
                (
                    "points0",
                    runTime.name(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE,
                    true
                ),
                pMesh,
                dimensionedVector(dimLength, Zero)
            )
        );

        if (have)
        {
            pointIOField points
            (
                IOobject
                (
                    "points",
                    runTime.constant(),
                    polyMesh::meshSubDir,
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE,
                    false
                )
            );

            if (points.size() != mesh.nPoints())
            {
                FatalErrorInFunction
                    << "Region " << regionName << ": constant points ("
                    << points.size() << ") do not match the mesh ("
                    << mesh.nPoints() << " points) on processor "
                    << Pstream::myProcNo()
                    << "; cannot transport the motion reference"
                    << exit(FatalError);
            }

            p0().primitiveFieldRef() = points;
        }
    }

    pointVectors.append(p0.ptr());
}


// Re-create constant/<region>/polyMesh/points in the written layout
// from the transported reference. Runs after mesh.write(); no-op when
// dmrEnsurePoints0 did not register a reference
static void dmrWritePoints0Constant(const fvMesh& mesh)
{
    if (!mesh.foundObject<pointVectorField>("points0"))
    {
        return;
    }

    const pointVectorField& p0 =
        mesh.lookupObject<pointVectorField>("points0");

    pointIOField points
    (
        IOobject
        (
            "points",
            mesh.time().constant(),
            polyMesh::meshSubDir,
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        p0.primitiveField()
    );

    points.write();
}


// fvMeshDistribute requires an internal-type patch when fields are
// registered (findInternalPatch). The on-disk case must not carry one
// (it clashes with the non-conformal solver machinery), so a transient
// zero-size internalPolyPatch is added to the private mesh copy after
// field loading (fvMesh::addPatch extends every registered field's
// boundary) and removed again before writing. Same insertion pattern
// as createNonConformalCouples.
static const word dmrTransientPatchName("dmrTransientInternal");

static bool dmrAddTransientInternalPatch(fvMesh& mesh)
{
    const polyBoundaryMesh& pbm = mesh.poly().boundary();

    forAll(pbm, patchi)
    {
        if (isA<internalPolyPatch>(pbm[patchi]))
        {
            return false;
        }
    }

    // Zero-size, positioned where fvMeshTools::addPatch inserts it:
    // before the first processor patch, or at the end.
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
            dmrTransientPatchName,
            0,
            start,
            pbm.size(),
            pbm
        )
    );

    return true;
}


// Remove the transient patch before writing. It must be empty again
// (exposed faces are re-homed onto processor patches by the end of
// distribute); the check is reduced so all ranks take the same branch,
// since reorderPatches is collective.
static void dmrRemoveTransientInternalPatch(fvMesh& mesh)
{
    const polyBoundaryMesh& pbm = mesh.poly().boundary();
    const label patchi = pbm.findIndex(dmrTransientPatchName);

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


// Source the shrink-side redistribute from the live mesh in memory
// rather than from the checkpoint on disk. Off by default while the
// clone path is being validated against the disk path; set
// FOAM_DMR_CLONE=1 to enable.
static bool dmrCloneEnabled()
{
    static const bool enabled = []()
    {
        const char* env = std::getenv("FOAM_DMR_CLONE");
        return env && std::string(env) == "1";
    }();
    return enabled;
}


// In-memory twin of the disk fresh-load: build a clean, non-registered
// fvMesh from the LIVE solver mesh's primitives instead of re-reading
// what we just checkpointed. Same construction fvMeshDistribute itself
// uses for received meshes (fvMeshDistribute.C receiveMesh: from-
// primitives fvMesh + polyPatch list + addFvPatches, no parallel
// comms), and the same primitives its sendMesh transmits
// (points/faces/faceOwner/faceNeighbour), so allOwner is per-face and
// allNeighbour per-internal-face.
//
// register=false and the real region name mirror the disk fresh-load
// exactly: the clone must not appear in runTime's registry beside the
// live mesh, but must still write to the region's own directory.
static autoPtr<fvMesh> dmrCloneLiveMesh
(
    const fvMesh& live,
    Time& runTime,
    const word& regionName
)
{
    pointField points(live.points());
    faceList faces(live.faces());
    labelList allOwner(live.faceOwner());
    labelList allNeighbour(live.faceNeighbour());

    autoPtr<fvMesh> clonePtr
    (
        new fvMesh
        (
            IOobject
            (
                regionName,
                runTime.name(),
                runTime,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE,
                false
            ),
            move(points),
            move(faces),
            move(allOwner),
            move(allNeighbour)
        )
    );
    fvMesh& clone = clonePtr();

    // Rebuild the boundary from the live patches. clone(bm) is virtual,
    // so each patch type copies its own state: processorPolyPatch keeps
    // myProcNo/neighbProcNo, cyclics keep their transform, etc. Sizes
    // and start faces carry over unchanged because the topology is
    // identical by construction.
    const polyBoundaryMesh& livePatches = live.poly().boundary();

    List<polyPatch*> patches(livePatches.size());
    forAll(livePatches, patchi)
    {
        patches[patchi] =
            livePatches[patchi].clone(clone.poly().boundary()).ptr();
    }

    // No parallel comms: the patch structure is already consistent
    // across ranks (it is a copy of the live mesh's, which is)
    clone.addFvPatches(patches, false);

    return clonePtr;
}


// Copy the live solver's checkpoint fields of one type onto the clone.
//
// ONLY the AUTO_WRITE set is taken, and that filter is the correctness
// invariant of the whole clone path: it is exactly what runTime.write()
// would have emitted, i.e. exactly what the disk fresh-load would have
// read back. The live registry additionally carries solver-derived and
// cached state (rAU, HbyA, gradients, functionObject fields, retained
// tmps) which is NO_WRITE and may differ between ranks;
// fvMeshDistribute enumerates registered fields through fieldNames()
// -> checkEqualWordList(), a gather/scatter-collective that FATALs
// unless every rank presents an identical list, so letting that state
// through is precisely what makes distribute() fail on a live mesh.
//
// Transport is a stream round-trip into the field's dictionary form and
// back through the GeoField(IOobject, mesh, dictionary) constructor —
// the same mechanism the grow path uses to rebuild fields on ranks that
// had none, and the same representation a disk write/read pair goes
// through. It is used in preference to a direct value copy because the
// from-components constructor clones patch fields via
// PatchField::clone(iF), which retains the SOURCE patch reference
// (GeometricBoundaryField.C): the fields would still point into the
// live mesh's boundary. Going through the dictionary rebuilds genuine
// patch fields on the clone's own patches, carrying each BC's full
// state (refValue, valueFraction, gradient, ...), not just its values.
template<class GeoField>
static void dmrCloneFields
(
    const fvMesh& live,
    const fvMesh& clone,
    const typename GeoField::Mesh& cloneFieldMesh,
    PtrList<GeoField>& fields
)
{
    const HashTable<const GeoField*> liveFields
    (
        live.lookupClass<GeoField>()
    );

    // sortedToc: every rank must build its list in the same order
    const wordList names(liveFields.sortedToc());

    DynamicList<word> checkpointed(names.size());
    forAll(names, i)
    {
        if (liveFields[names[i]]->writeOpt() == IOobject::AUTO_WRITE)
        {
            checkpointed.append(names[i]);
        }
    }

    fields.setSize(checkpointed.size());
    forAll(checkpointed, i)
    {
        const GeoField& src = *liveFields[checkpointed[i]];

        OStringStream os;
        os.precision(17);   // exact round-trip of an IEEE double
        os << src;

        IStringStream is(os.str());
        const dictionary fieldDict(is);

        fields.set
        (
            i,
            new GeoField
            (
                IOobject
                (
                    checkpointed[i],
                    clone.time().name(),
                    clone,
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                cloneFieldMesh,
                fieldDict
            )
        );
    }
}


// Source the fields either from the live mesh (clone path) or from the
// checkpoint on disk (fresh-load path), keeping the 15 call sites below
// single.
template<class GeoField>
static void dmrGatherFields
(
    const fvMesh* live,
    const fvMesh& mesh,
    const typename GeoField::Mesh& fieldMesh,
    IOobjectList& allObjects,
    PtrList<GeoField>& fields
)
{
    if (live)
    {
        dmrCloneFields(*live, mesh, fieldMesh, fields);
    }
    else
    {
        dmrLoadFields(fieldMesh, allObjects, fields);
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
    const word& regionName,
    const fvMesh* live
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

    // Either clone the live mesh in memory or re-read the checkpoint
    // from disk. Everything downstream is identical: the clone is built
    // to be indistinguishable from what the disk read produces.
    autoPtr<fvMesh> meshPtr;
    if (live)
    {
        meshPtr = dmrCloneLiveMesh(*live, runTime, regionName);
    }
    else
    {
        meshPtr.reset
        (
            new fvMesh
            (
                IOobject
                (
                    regionName,
                    runTime.findInstance(meshSubDir, "points"),
                    runTime,
                    IOobject::MUST_READ,
                    IOobject::AUTO_WRITE,
                    false
                )
            )
        );
    }
    fvMesh& mesh = meshPtr();

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

    dmrGatherFields(live, mesh, mesh, allObjects, volScalars);
    dmrGatherFields(live, mesh, mesh, allObjects, volVectors);
    dmrGatherFields(live, mesh, mesh, allObjects, volSphereTensors);
    dmrGatherFields(live, mesh, mesh, allObjects, volSymmTensors);
    dmrGatherFields(live, mesh, mesh, allObjects, volTensors);
    dmrGatherFields(live, mesh, mesh, allObjects, surfScalars);
    dmrGatherFields(live, mesh, mesh, allObjects, surfVectors);
    dmrGatherFields(live, mesh, mesh, allObjects, surfSphereTensors);
    dmrGatherFields(live, mesh, mesh, allObjects, surfSymmTensors);
    dmrGatherFields(live, mesh, mesh, allObjects, surfTensors);

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

    dmrGatherFields(live, mesh, pMesh, allObjects, pointScalars);
    dmrGatherFields(live, mesh, pMesh, allObjects, pointVectors);
    dmrGatherFields(live, mesh, pMesh, allObjects, pointSphereTensors);
    dmrGatherFields(live, mesh, pMesh, allObjects, pointSymmTensors);
    dmrGatherFields(live, mesh, pMesh, allObjects, pointTensors);

    dmrEnsurePoints0
    (
        runTime,
        mesh,
        pMesh,
        boolList(),
        autoPtr<fvMeshSubset>(),
        regionName,
        pointVectors
    );

    const bool addedPatch = dmrAddTransientInternalPatch(mesh);

    fvMeshDistribute(mesh).distribute(finalDecomp);

    if (addedPatch)
    {
        dmrRemoveTransientInternalPatch(mesh);
    }

    mesh.setInstance(timeName);
    mesh.write();

    dmrWritePoints0Constant(mesh);
}


// Sweep the fvMeshStitcher caches (fvMesh/polyFaces, a label surface
// field fvMeshDistribute cannot transport) from this rank's processor
// directory, at both the given time and the constant instance. The
// cache is derivable: when absent (READ_IF_PRESENT), the stitcher
// reconnects from geometry at the next startup. Must run only after
// ALL regions have been redistributed: sibling regions re-read each
// other's caches through mapped/NCC patches during fresh-load, so a
// per-region sweep starves the next region's construction. No-op for
// non-NCC cases.
static void dmrSweepStitchCaches
(
    Time& runTime,
    const std::string& timeName,
    const wordList& regionNames
)
{
    forAll(regionNames, i)
    {
        const fileName regionDir =
            regionNames[i] == polyMesh::defaultRegion
          ? fileName(".")
          : fileName(regionNames[i]);

        const fileName cacheT =
            runTime.path()/timeName/regionDir/"fvMesh";
        const fileName cacheC =
            runTime.path()/runTime.constant()/regionDir/"fvMesh";

        if (isDir(cacheT)) rmDir(cacheT);
        if (isDir(cacheC)) rmDir(cacheC);

        // tetBasePtIs is the same class of derivable cache, with a
        // sharper failure mode: a generation that ran sampling-type
        // functionObjects allocates it, so its checkpoint writeNow
        // WRITES it (polyMeshIO propagates writeOpt); the fresh-load
        // re-reads it (READ_IF_PRESENT in the polyMesh ctor),
        // fvMeshDistribute never remaps it (0 references), and
        // mesh.write() re-emits STALE OLD-LAYOUT tet data — but only
        // on the ranks that had data, so at restart those ranks skip
        // the collective findFaceBasePts that the others enter:
        // off-by-N collective desync (movingCone, 2026-07-13).
        // Absent, every rank recomputes it collectively and
        // consistently.
        const fileName meshDirT =
            runTime.path()/timeName/regionDir/"polyMesh";
        const fileName meshDirC =
            runTime.path()/runTime.constant()/regionDir/"polyMesh";

        rm(meshDirT/"tetBasePtIs");
        rm(meshDirC/"tetBasePtIs");
    }
}


// Swap-mesh transport (fvMeshTopoChangers::meshToMesh). The topo
// changer reads constant/meshes/<meshTime>[/<region>]/polyMesh per
// rank, lazily, at each swap time, and mesh().swap() makes the swap
// mesh's decomposition BECOME the solver decomposition; after a
// reconfiguration the pieces must therefore exist, complete, in the
// current world size (ranks added by a grow have none, and after a
// shrink the pieces beyond the new world size would silently drop
// their cells at the next swap).
//
// The pieces are RE-DERIVED from the serial originals rather than
// redistributed: constant/meshes/<t> at the CASE ROOT is static
// reference data that survives every reconfiguration, so each rank
// forks the stock `decomposePar -mesh <t> [-region <r>]` for a
// round-robin share of the meshes, producing pieces for the current
// world size (numberOfSubdomains was already rewritten for this
// reconfiguration) that are byte-equivalent to the original case
// preparation. Embarrassingly parallel across meshes; serial per
// mesh, acceptable for read-only reference data.
//
// This replaces an fvMeshDistribute-based transport: distributing
// the old pieces failed with "The ordering walk did not hit every
// face exactly once" in processorPolyPatch::order on a checkMesh-
// clean engine swap mesh (the merge of received pieces can leave the
// two sides of a new processor interface with inconsistent edge
// connectivity; upstream-report material, see doc 4).
//
// decomposePar's -force flag is NEVER used here: it removes whole
// processor trees, including the redistributed solution. Stale
// pieces are removed surgically instead.
static void dmrRedistributeSwapMeshes(Time& runTime)
{
    const std::string casePath = runTime.globalPath();

    // The master enumerates the SERIAL originals at the case root
    wordList meshTimes;
    wordList meshRegions;   // empty word: default-region layout
    if (Pstream::master())
    {
        const fileName meshesDir =
            fileName(casePath)/runTime.constant()/"meshes";

        if (isDir(meshesDir))
        {
            const fileNameList times
            (
                readDir(meshesDir, fileType::directory)
            );

            DynamicList<word> ts;
            DynamicList<word> rs;
            forAll(times, i)
            {
                if (isDir(meshesDir/times[i]/polyMesh::meshSubDir))
                {
                    ts.append(word(times[i]));
                    rs.append(word::null);
                }
                else
                {
                    const fileNameList regions
                    (
                        readDir(meshesDir/times[i], fileType::directory)
                    );
                    forAll(regions, j)
                    {
                        if
                        (
                            isDir
                            (
                                meshesDir/times[i]/regions[j]
                               /polyMesh::meshSubDir
                            )
                        )
                        {
                            ts.append(word(times[i]));
                            rs.append(word(regions[j]));
                        }
                    }
                }
            }

            meshTimes.transfer(ts);
            meshRegions.transfer(rs);
        }
    }
    Pstream::scatter(meshTimes);
    Pstream::scatter(meshRegions);

    if (meshTimes.empty())
    {
        return;
    }

    // Master-serial, mirroring the thesis legacy path (rank 0 forks
    // stock decomposePar per swap mesh). Concurrent forks across ranks
    // race on the shared case directory (all read the processor-dir
    // count and decomposeParDict); serialising on the master removes
    // that hazard, and the swap meshes are small static reference data.
    //
    // decomposePar refuses ("Case is already decomposed with N domains")
    // whenever the case-level processor* count differs from
    // numberOfSubdomains. During a shrink the CALLER must therefore have
    // already deleted the processor dirs beyond the new world size
    // (dmrShrinkInPlace removes processor[nNew..nOld) before calling us),
    // so the surviving count equals the just-rewritten numberOfSubdomains
    // and the guard passes without -force (which would nuke the whole
    // redistributed solution). Grow needs no such trim: the grow
    // redistribute already populated exactly nNew processor dirs.
    if (Pstream::master())
    {
        // Clear any stale pieces in the surviving processor dirs so
        // decomposePar writes a clean decomposition
        for (label proci = 0; proci < Pstream::nProcs(); proci++)
        {
            const fileName pieces =
                fileName(casePath)/("processor" + Foam::name(proci))
               /runTime.constant()/"meshes";
            if (isDir(pieces))
            {
                rmDir(pieces);
            }
        }

        forAll(meshTimes, meshi)
        {
            std::string cmd =
                "decomposePar -mesh '" + std::string(meshTimes[meshi])
              + "' -case '" + casePath + "'";
            if (!meshRegions[meshi].empty())
            {
                cmd += " -region '" + std::string(meshRegions[meshi]) + "'";
            }

            dmrRunOrAbort(casePath, "dmr_decompose_meshes.log", cmd);
        }

        Info<< "DMR: re-decomposed " << meshTimes.size()
            << " swap mesh(es) under constant/meshes for "
            << Pstream::nProcs() << " ranks" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void dmrGrowRedistribute(Time& runTime, bool allRegions)
{
    // Only a process generation that follows a DMR GROW reconfiguration
    // has work to do here. The reconfig count guards fresh launches
    // against stale meta files; the meta is renamed once consumed.
    if (dmr_get_reconfig_count() <= 0)
    {
        return;
    }

    const std::string casePath = runTime.globalPath();

    DmrWorldInfo world;
    if (!dmrReadWorldMeta(casePath, world) || world.dir != DmrDir::Grow)
    {
        return;
    }

    const std::string timeName = runTime.name();

    if (Pstream::master())
    {
        dmrLifecycleLog
        (
            casePath,
            "Grow redistribute at t=" + timeName
          + ": inflating " + std::to_string(world.nOld)
          + "-way layout to " + std::to_string(world.nNew) + " ranks."
        );
    }

    // Not all ranks have meshes yet; master-only reading cannot work.
    regIOobject::fileModificationChecking = regIOobject::timeStamp;

    wordList regionNames;
    if (allRegions)
    {
        const std::string pattern =
            casePath + "/processor0/constant/*/polyMesh";
        glob_t g;
        if (glob(pattern.c_str(), GLOB_ONLYDIR, nullptr, &g) == 0)
        {
            regionNames.setSize(g.gl_pathc);
            forAll(regionNames, i)
            {
                const fileName p(g.gl_pathv[i]);
                regionNames[i] = p.path().name();
            }
            globfree(&g);
        }
    }
    else
    {
        regionNames = wordList(1, word(polyMesh::defaultRegion));
    }

    forAll(regionNames, regioni)
    {
        const word& regionName = regionNames[regioni];

        fileName meshSubDir;
        if (regionName == polyMesh::defaultRegion)
        {
            meshSubDir = polyMesh::meshSubDir;
        }
        else
        {
            meshSubDir = regionName / polyMesh::meshSubDir;
        }

        // Only the master is guaranteed to have the mesh on disk
        fileName masterInstDir;
        if (Pstream::master())
        {
            masterInstDir = runTime.findInstance(meshSubDir, "points");
        }
        Pstream::scatter(masterInstDir);

        boolList haveMesh(Pstream::nProcs(), false);
        haveMesh[Pstream::myProcNo()] =
            isDir(runTime.path()/masterInstDir/meshSubDir);
        Pstream::gatherList(haveMesh);
        Pstream::scatterList(haveMesh);

        autoPtr<fvMesh> meshPtr = loadOrCreateMesh
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
        fvMesh& mesh = meshPtr();

        // loadOrCreateMesh's dummy zones must never reach the written
        // mesh: the unoriented dummyFaceZone FATALs in a later
        // fvMeshDistribute zone sync (flipMap). The utility's own sync
        // clears them, but they have been observed to survive into the
        // write; purge by name. Dummies never coexist with real zones
        // (the sync replaces them when real zones exist), so this is a
        // no-op on every legitimate case.
        if (mesh.faceZones().findIndex("dummyFaceZone") != -1)
        {
            Pout<< "DMR-GROW[r" << Pstream::myProcNo()
                << "] purging dummy zones after loadOrCreateMesh" << endl;
            mesh.pointZones().clear();
            mesh.faceZones().clear();
            mesh.cellZones().clear();
        }

        // Zero-cell subsetter so the master can send zero-sized fields
        // to the ranks that have none (redistributePar.C pattern)
        autoPtr<fvMeshSubset> subsetterPtr;

        bool allHaveMesh = true;
        forAll(haveMesh, proci)
        {
            if (!haveMesh[proci])
            {
                allHaveMesh = false;
                break;
            }
        }

        if (!allHaveMesh)
        {
            const polyBoundaryMesh& patches = mesh.poly().boundary();

            label nonProci = -1;
            forAll(patches, patchi)
            {
                if (isA<processorPolyPatch>(patches[patchi]))
                {
                    break;
                }
                nonProci++;
            }

            subsetterPtr.reset(new fvMeshSubset(mesh));
            subsetterPtr().setLargeCellSubset
            (
                labelHashSet(0),
                nonProci,
                false
            );

            // fvMeshSubset::interpolate on a point field lazily
            // constructs the subset mesh's pointMesh, whose
            // constructor (pointBoundaryMesh::calcGeometry) is
            // COLLECTIVE — but interpolate is only called on the
            // master (zero-field sends), deadlocking every other rank
            // in the following scatter. Stock redistributePar has the
            // same defect whenever point fields are present (gdb
            // stacks, movingCone, 2026-07-12). Constructing the
            // pointMesh collectively up front makes the master-only
            // path a pure cache hit.
            (void)pointMesh::New(subsetterPtr().subMesh());
        }

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

        dmrLoadFieldsGrow(haveMesh, mesh, subsetterPtr, allObjects, volScalars);
        dmrLoadFieldsGrow(haveMesh, mesh, subsetterPtr, allObjects, volVectors);
        dmrLoadFieldsGrow(haveMesh, mesh, subsetterPtr, allObjects, volSphereTensors);
        dmrLoadFieldsGrow(haveMesh, mesh, subsetterPtr, allObjects, volSymmTensors);
        dmrLoadFieldsGrow(haveMesh, mesh, subsetterPtr, allObjects, volTensors);
        dmrLoadFieldsGrow(haveMesh, mesh, subsetterPtr, allObjects, surfScalars);
        dmrLoadFieldsGrow(haveMesh, mesh, subsetterPtr, allObjects, surfVectors);
        dmrLoadFieldsGrow(haveMesh, mesh, subsetterPtr, allObjects, surfSphereTensors);
        dmrLoadFieldsGrow(haveMesh, mesh, subsetterPtr, allObjects, surfSymmTensors);
        dmrLoadFieldsGrow(haveMesh, mesh, subsetterPtr, allObjects, surfTensors);

        pointMesh& pMesh = const_cast<pointMesh&>(pointMesh::New(mesh));

        PtrList<pointScalarField>          pointScalars;
        PtrList<pointVectorField>          pointVectors;
        PtrList<pointSphericalTensorField> pointSphereTensors;
        PtrList<pointSymmTensorField>      pointSymmTensors;
        PtrList<pointTensorField>          pointTensors;

        dmrLoadFieldsGrow(haveMesh, pMesh, subsetterPtr, allObjects, pointScalars);
        dmrLoadFieldsGrow(haveMesh, pMesh, subsetterPtr, allObjects, pointVectors);
        dmrLoadFieldsGrow(haveMesh, pMesh, subsetterPtr, allObjects, pointSphereTensors);
        dmrLoadFieldsGrow(haveMesh, pMesh, subsetterPtr, allObjects, pointSymmTensors);
        dmrLoadFieldsGrow(haveMesh, pMesh, subsetterPtr, allObjects, pointTensors);

        dmrEnsurePoints0
        (
            runTime,
            mesh,
            pMesh,
            haveMesh,
            subsetterPtr,
            regionName,
            pointVectors
        );

        const bool addedPatch = dmrAddTransientInternalPatch(mesh);

        autoPtr<decompositionMethod> distributor
        (
            decompositionMethod::NewDistributor
            (
                decompositionMethod::decomposeParDict(runTime)
            )
        );
        const labelList finalDecomp =
            distributor().decompose(mesh, mesh.cellCentres());

        fvMeshDistribute(mesh).distribute(finalDecomp);

        if (addedPatch)
        {
            dmrRemoveTransientInternalPatch(mesh);
        }

        // Second purge site: catches dummies resurrected inside
        // distribute (zone merge across ranks).
        if (mesh.faceZones().findIndex("dummyFaceZone") != -1)
        {
            Pout<< "DMR-GROW[r" << Pstream::myProcNo()
                << "] purging dummy zones after distribute" << endl;
            mesh.pointZones().clear();
            mesh.faceZones().clear();
            mesh.cellZones().clear();
        }

        mesh.setInstance(timeName);
        mesh.write();

        dmrWritePoints0Constant(mesh);

        // If the in-memory zone lists are empty, any *Zones files at
        // the write instance are stale leftovers (the dummy mesh from
        // loadOrCreateMesh writes its dummy zones into the same dir
        // when the staged instance coincides with the write time, and
        // its cleanup does not always take them out). An unoriented
        // dummyFaceZone read back later FATALs in fvMeshDistribute.
        if
        (
            mesh.pointZones().empty()
         && mesh.faceZones().empty()
         && mesh.cellZones().empty()
        )
        {
            const fileName meshDir =
                runTime.path()/timeName/meshSubDir;

            rm(meshDir/"pointZones");
            rm(meshDir/"faceZones");
            rm(meshDir/"cellZones");
        }

        // loadOrCreateMesh wrote a dummy zero-cell mesh (with dummy
        // point/face/cell zones) at the staged instance on the ranks
        // that had no data. The real mesh now lives at timeName, but
        // the dummy files persist and a later fresh-load's zone
        // findInstance walks back into them (unoriented dummyFaceZone
        // FATALs in fvMeshDistribute's zone sync). Remove them; if the
        // staged instance IS the write time, remove only the dummy
        // zone files beside the real mesh.
        if (!haveMesh[Pstream::myProcNo()])
        {
            const fileName dummyDir =
                runTime.path()/masterInstDir/meshSubDir;

            if (fileName(masterInstDir) != fileName(timeName))
            {
                if (isDir(dummyDir)) rmDir(dummyDir);
            }
            else
            {
                rm(dummyDir/"pointZones");
                rm(dummyDir/"faceZones");
                rm(dummyDir/"cellZones");
            }
        }
    }

    // Sweep the derivable caches BEFORE the swap-mesh transport: if
    // the latter fails, the relaunched generation must never read a
    // stale mixed-layout stitcher cache (engine, 2026-07-13)
    dmrSweepStitchCaches(runTime, timeName, regionNames);

    dmrRedistributeSwapMeshes(runTime);

    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        // Consume the meta so a later fresh launch of this case (or the
        // next generation) cannot mistake it for an in-flight grow
        std::rename
        (
            (casePath + "/log.dmr/.world.meta").c_str(),
            (casePath + "/log.dmr/.world.meta.done").c_str()
        );
        ::sync();

        dmrLifecycleLog
        (
            casePath,
            "Grow redistribute complete ("
          + std::to_string(regionNames.size()) + " region(s))."
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);
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

    const double redistStart = MPI_Wtime();

    forAll(regionNames, i)
    {
        // The live mesh is still resident on this (dying) group, so the
        // checkpoint it was just written from can be reproduced in
        // memory instead of being read back off disk
        const fvMesh* live =
            dmrCloneEnabled()
          ? &runTime.lookupObject<fvMesh>(regionNames[i])
          : nullptr;

        dmrRunFreshRedistribute(runTime, timeName, regionNames[i], live);
    }

    const double redistEnd = MPI_Wtime();

    // Sweep the derivable caches BEFORE the swap-mesh transport: if
    // the latter fails, the relaunched generation must never read a
    // stale mixed-layout stitcher cache (engine, 2026-07-13)
    dmrSweepStitchCaches(runTime, timeName, regionNames);

    // Delete the processor dirs beyond the new world size BEFORE the
    // swap-mesh decompose. dmrRedistributeSwapMeshes forks stock
    // decomposePar, which FATALs unless the case-level processor count
    // equals numberOfSubdomains (just rewritten to nNew); the stale
    // processor[nNew..nOld) from the old layout would trip that guard
    // (engine/movingCone swap-mesh decompose, 2026-07-18).
    MPI_Barrier(MPI_COMM_WORLD);
    if (Pstream::master())
    {
        for (int p = world.nNew; p < world.nOld; ++p)
        {
            Foam::rmDir(casePath + "/processor" + std::to_string(p));
        }
        ::sync();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    dmrRedistributeSwapMeshes(runTime);

    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        char elapsed[32];
        std::snprintf(elapsed, sizeof(elapsed), "%.3f", redistEnd - redistStart);

        dmrLifecycleLog
        (
            casePath,
            "Shrink redistribute complete ("
          + std::to_string(regionNames.size()) + " region(s)) in "
          + elapsed + " s ["
          + (dmrCloneEnabled() ? "clone" : "disk") + "]."
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

    if (worldOk && world.dir == DmrDir::Grow)
    {
        // The NEW group has the max(N_old, N_new) ranks, so the data
        // move happens there (dmrGrowRedistribute, hooked after
        // createTime). The dying group only flushes its state.
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
            dmrWriteWorldMeta(casePath, world);
            dmrLifecycleLog
            (
                casePath,
                "Reconfig GROW: N_old=" + std::to_string(world.nOld)
              + " -> N_new=" + std::to_string(world.nNew)
              + " (redistribute deferred to the new group)."
            );
            ::sync();
        }

        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    // Unknown direction: legacy reconstructPar path (safety net).
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

        // NewDistributor (used by the in-process redistributes) reads
        // its own 'distributor' key, separate from 'decomposer'.
        dmrUpdateDictEntry
        (
            casePath, "system/decomposeParDict",
            "distributor", "scotch"
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
        else if (worldOk && world.dir == DmrDir::Grow)
        {
            dmrLifecycleLog
            (
                casePath,
                "Grow restart: staging empty time dirs on the new ranks;"
                " redistribute runs in the solver hook."
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

    // GROW pre-step, collective: ranks beyond the old world have no
    // processor directory yet, and Time::setControls FATALs unless
    // every rank resolves the same startFrom-latestTime instant. Each
    // new rank stages its processor dir with an EMPTY restart-time
    // dir; mesh and fields arrive later via dmrGrowRedistribute.
    if (worldOk && world.dir == DmrDir::Grow)
    {
        const std::string t = dmrLatestProcTimeName(casePath);

        if (t.empty())
        {
            std::fprintf
            (
                stderr,
                "DMR: FATAL — no time directories in '%s/processor0'.\n",
                casePath.c_str()
            );
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (myRank >= world.nOld)
        {
            Foam::mkDir
            (
                casePath + "/processor" + std::to_string(myRank) + "/" + t
            );
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
