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
#include "fvMeshStitcher.H"
#include "nonConformalFvPatch.H"
#include "mixedFvPatchField.H"
#include "fixedGradientFvPatchField.H"
#include "surfaceInterpolate.H"
#include "surfaceToVolVelocity.H"
#include "nullObject.H"
#include "nonConformalMappedPolyFacesFvsPatchLabelField.H"
#include "nonConformalProcessorCyclicPolyPatch.H"
#include "solver.H"
#include "IOPtrList.H"
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
#include "IFstream.H"
#include "OStringStream.H"
#include "IStringStream.H"
#include "CompactListList.H"

extern "C"
{
    #include "dmr.h"
}

#include <mpi.h>
#include <glob.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <limits>
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
            // Not a numeric dir
        }
    }
    globfree(&g);
    return maxName;
}


// Largest numeric time in processor0/. Checkpoints exist only in the
// processor layout, so this, not dmrLatestTimeName, names the restart
// time.
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


// Rank 0 queries DMR analytics and the procs_next_* getters and
// broadcasts the result. Only valid inside the DMR_AUTO redistribution
// callback (after dmr_reconfigure returns DMR_REDIST_FINALIZE, before
// dmr_finalize).
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
        const DMRStatus st = dmr_get_analytics(&a);

        if (st == DMR_SUCCESS)
        {
            // DMRAnalytics::event is an enum in older DMR and a string in
            // newer, which also shifts every later field: compiling
            // against one and linking the other corrupts the struct
            // silently. The string API #defines the event names, which
            // tells the two apart at compile time.
            #ifdef DMR_EVENT_START_SHRINK
            // String API (dmr-s4d and later)
            const char* ev = a.event ? a.event : "";

            if
            (
                std::strcmp(ev, DMR_EVENT_START_EXPAND_SLURM) == 0
             || std::strcmp(ev, DMR_EVENT_START_EXPAND_MPI) == 0
            )
            {
                dir = static_cast<int>(DmrDir::Grow);
                delta = dmr_get_procs_next_expand();
            }
            else if (std::strcmp(ev, DMR_EVENT_START_SHRINK) == 0)
            {
                dir = static_cast<int>(DmrDir::Shrink);
                delta = dmr_get_procs_next_shrink();
            }
            #else
            // Enum API (dmr-at-jobs 2.0.1 and earlier)
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
            #endif
        }

        // The direction must be known here: a shrink runs on the old
        // (larger) group, since fvMeshDistribute needs max(N, N') ranks.
        // If it is Unknown the run stops, so report whether the analytics
        // call failed or returned an event not handled above (the CE
        // policy emits some the ROUND policy never does).
        if (dir == static_cast<int>(DmrDir::Unknown))
        {
            #ifdef DMR_EVENT_START_SHRINK
            const char* evStr = a.event ? a.event : "(null)";
            #else
            char evBuf[32];
            std::snprintf(evBuf, sizeof(evBuf), "%d", static_cast<int>(a.event));
            const char* evStr = evBuf;
            #endif

            // A struct-layout mismatch shows up here as a nonsense
            // world_size with node_count holding its value instead.
            std::fprintf
            (
                stderr,
                "DMR: direction UNKNOWN — dmr_get_analytics status=%d,"
                " event='%s', world_size=%d, node_count=%d,"
                " pending_nodes=%d, CE=%f\n",
                static_cast<int>(st), evStr,
                a.world_size, a.node_count, a.pending_nodes,
                a.communication_efficiency
            );
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


// Grow-side field reader, after redistributePar.C's readFields, on the
// ranks that hold a mesh. The read happens stitched, in the sizes the
// checkpoint has. The ranks the grow added get zero-sized copies later
// (dmrSendZeroFieldsGrow), once every region is conformal again, because
// fvMeshSubset::interpolate needs fvPatch::start(), which a
// non-conformal patch with faces does not have.
template<class GeoField>
static void dmrReadFieldsGrow
(
    const boolList& haveMesh,
    const typename GeoField::Mesh& mesh,
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

    if (!haveMesh[Pstream::myProcNo()])
    {
        return;
    }

    fields.setSize(masterNames.size());

    forAll(masterNames, i)
    {
        IOobject& io = objects[masterNames[i]];
        io.writeOpt() = IOobject::AUTO_WRITE;

        fields.set(i, new GeoField(io, mesh));
    }
}


// The registry a field of the given mesh type lives in
static const objectRegistry& dmrFieldDb(const fvMesh& mesh)
{
    return mesh;
}

static const objectRegistry& dmrFieldDb(const pointMesh& mesh)
{
    return mesh();
}


// The zero-sized copies for the ranks without a mesh, from the master's
// registered fields, which are conformal by now. The copies are stored
// in the receiving mesh's registry, as the read fields are.
template<class GeoField>
static void dmrSendZeroFieldsGrow
(
    const boolList& haveMesh,
    const typename GeoField::Mesh& mesh,
    const fvMeshSubset& subsetter
)
{
    const objectRegistry& db = dmrFieldDb(mesh);

    // Checkpoint fields only (dmrReadFieldsGrow marks them AUTO_WRITE).
    // The registry may also hold cached geometry (sliced, NO_WRITE fields
    // such as magSf) that a subset cannot interpolate.
    wordList names;
    if (Pstream::master())
    {
        const HashTable<const GeoField*> all(db.lookupClass<GeoField>());

        DynamicList<word> writable;
        forAllConstIter(typename HashTable<const GeoField*>, all, iter)
        {
            if (iter()->writeOpt() == IOobject::AUTO_WRITE)
            {
                writable.append(iter.key());
            }
        }

        names.transfer(writable);
        sort(names);
    }
    Pstream::scatter(names);

    if (Pstream::master())
    {
        forAll(names, i)
        {
            tmp<GeoField> tsubfld =
                subsetter.interpolate(db.lookupObject<GeoField>(names[i]));

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
    else if (!haveMesh[Pstream::myProcNo()])
    {
        forAll(names, i)
        {
            IPstream fromMaster
            (
                Pstream::commsTypes::blocking,
                Pstream::masterNo()
            );
            dictionary fieldDict(fromMaster);

            regIOobject::store
            (
                new GeoField
                (
                    IOobject
                    (
                        names[i],
                        db.time().name(),
                        db,
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE
                    ),
                    mesh,
                    fieldDict
                )
            );
        }
    }
}


// Whether anything can read this region's motion reference at startup.
// The readers of constant/<region>/polyMesh/points (or of points0, via
// points0MotionSolver::readPoints0) are the points0MotionSolver family,
// componentDisplacementMotionSolver, the inkJet and interpolator movers
// and the rigidBodyMotionDisplacement point condition. Rather than list
// them, this lists the movers that do not read it, selected the way
// fvMeshMover::New and motionSolver::New select them, and assumes a
// reader for anything else.
static bool dmrMoverReadsReference(const fileName& dynamicMeshDict)
{
    IFstream is(dynamicMeshDict);
    const dictionary dict(is);

    // No mover: fvMeshMovers::none
    if (!dict.found("mover"))
    {
        return false;
    }

    const dictionary& moverDict = dict.subDict("mover");
    const word moverType(moverDict.lookup("type"));

    if (moverType == "multiValveEngine")
    {
        return false;
    }

    if (moverType == "motionSolver")
    {
        const word solverType(moverDict.lookup("motionSolver"));

        return
            solverType != "velocityLaplacian"
         && solverType != "velocityComponentLaplacian";
    }

    return true;
}


// The one reader that is a boundary condition rather than a mover, so it
// can appear whatever the mover is
static bool dmrHasRigidBodyDisplacementPatch(const fvMesh& mesh)
{
    const HashTable<const pointVectorField*> fields
    (
        mesh.lookupClass<pointVectorField>()
    );

    forAllConstIter(HashTable<const pointVectorField*>, fields, iter)
    {
        const pointVectorField::Boundary& bf = iter()->boundaryField();

        forAll(bf, patchi)
        {
            if (bf[patchi].type() == "rigidBodyMotionDisplacement")
            {
                return true;
            }
        }
    }

    return false;
}


// Reference-configuration transport. points0MotionSolver and its family
// rebuild their reference points at startup via readPoints0 (a points0
// field at a time instance, else constant/<region>/polyMesh/points).
// Neither survives a redistribution: the constant mesh stays in the old
// layout, or is missing on ranks a grow added, and the size check in
// points0MotionSolver fails. So resolve the reference the way
// readPoints0 does and register it as a point field, which
// fvMeshDistribute then transports. dmrWritePoints0Constant re-creates
// the constant points in the new layout afterwards, the only place
// readPoints0 looks for non-default regions.
//
// Only for regions where something reads the reference. Elsewhere the
// constant points may legitimately describe another mesh: a
// mesh-to-mesh swap replaces the mesh but not constant/polyMesh/points.
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

    // The grow path hands its read fields to the registry before this
    if (mesh.foundObject<pointVectorField>("points0"))
    {
        return;
    }

    const fileName regionDir =
        regionName == polyMesh::defaultRegion
      ? fileName(".")
      : fileName(regionName);

    // Only regions whose reference something reads; the master decides.
    // dynamicMeshDict exists at case level only (decomposePar never
    // copies dictionaries into processor directories), so it is looked up
    // under globalPath, not path.
    bool moving = false;
    word inst;
    if (Pstream::master())
    {
        const fileName dynamicMeshDict
        (
            fileName(runTime.globalPath())
           /runTime.constant()/regionDir/"dynamicMeshDict"
        );

        moving =
            isFile(dynamicMeshDict)
         && (
                dmrMoverReadsReference(dynamicMeshDict)
             || dmrHasRigidBodyDisplacementPatch(mesh)
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
        // An earlier write left a points0 field in the current layout.
        // AUTO_WRITE, so readPoints0 still finds it after the
        // reconfiguration.
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
        // First reconfiguration: build it from the per-rank constant
        // points (readPoints0's own fallback). NO_WRITE: persisted only
        // through dmrWritePoints0Constant.
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


// fvMeshDistribute needs an internal-type patch to park exposed faces
// on (findInternalPatch). The case on disk must not carry one, so a
// zero-size internalPolyPatch is added to the private mesh copy for the
// distribution and removed before writing, inserted the way
// createNonConformalCouples does.
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


// Run the distributor and return one destination per cell.
//
// ptscotch returns a one-entry list for a zero-cell mesh, which
// fvMeshDistribute rejects ("Size of distribution:1 mesh nCells:0").
// Zero-cell meshes are normal on a grow, on the ranks that hold no data
// yet. decompose() is collective, so the empty ranks must still call it;
// the result is corrected afterwards instead.
static labelList dmrDecompose
(
    decompositionMethod& distributor,
    const fvMesh& mesh
)
{
    labelList decomp(distributor.decompose(mesh, mesh.cellCentres()));

    if (mesh.nCells() == 0 && decomp.size() != 0)
    {
        Pout<< "DMR: distributor returned " << decomp.size()
            << " entries for a 0-cell mesh; truncating to 0" << endl;

        decomp.setSize(0);
    }

    return decomp;
}


// Partitioning method for the in-process redistributions. ptscotch is
// the parallel partitioner; serial scotch does not scale (a 100M-cell
// case sat in its decompose() for over an hour).
static const char* const dmrDistributorName = "ptscotch";


// Build a non-registered fvMesh from the live solver mesh's primitives,
// instead of re-reading what was just checkpointed. Same construction
// fvMeshDistribute::receiveMesh uses. Not registered, so it does not sit
// in runTime's registry beside the live mesh, but named after the region
// so it writes to the region's directory.
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

    // Rebuild the boundary from the live patches; clone() keeps each
    // patch type's own state (processor neighbours, cyclic transforms).
    // Every rank clones its own live mesh, so the patches' cached state
    // is consistent across ranks, which the collective below relies on
    // (contrast dmrRedistributeSwapMesh).
    const polyBoundaryMesh& livePatches = live.poly().boundary();

    List<polyPatch*> patches(livePatches.size());
    forAll(livePatches, patchi)
    {
        patches[patchi] =
            livePatches[patchi].clone(clone.poly().boundary()).ptr();
    }

    // validBoundary true (collective topoChange and calcGeometry) is
    // required: a cloned processorPolyPatch has empty neighbour geometry
    // until topoChange exchanges it, and a cyclic-type patch field
    // evaluates at construction, computing weights for every patch from
    // that geometry. receiveMesh can pass false because it maps fields
    // instead of constructing them.
    clone.addFvPatches(patches, true);

    return clonePtr;
}


// Copy the live solver's checkpoint fields of one type onto the clone.
//
// Only AUTO_WRITE fields, i.e. exactly what a checkpoint write would
// emit. The live registry also holds solver state (rAU, HbyA, cached
// gradients...) that may differ between ranks, and fvMeshDistribute
// fails unless every rank presents the same field list.
//
// Copied through the field's dictionary form rather than by value: a
// value copy clones the patch fields with clone(iF), which keeps the
// live mesh's patch reference. The dictionary rebuilds them on the
// clone's patches with each condition's full state.
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


// Take the checkpoint fields of one type out of the reconfiguration
// write, remembering them so they can be put back.
template<class GeoField>
static void dmrDemoteFieldsOfType
(
    const fvMesh& mesh,
    DynamicList<regIOobject*>& demoted
)
{
    const HashTable<const GeoField*> fields(mesh.lookupClass<GeoField>());
    const wordList names(fields.sortedToc());

    forAll(names, i)
    {
        // Restored right after the write; only the write option changes
        GeoField& f = const_cast<GeoField&>(*fields[names[i]]);

        if (f.writeOpt() == IOobject::AUTO_WRITE)
        {
            f.writeOpt() = IOobject::NO_WRITE;
            demoted.append(&f);
        }
    }
}


// The redistribution writes the fields in the new layout, so writing
// them in the old layout at the checkpoint as well is wasted I/O. The
// checkpoint itself cannot be skipped (it writes uniform/time, among
// others), so the fields are taken out of it: demote, write, restore.
//
// Consequence: if the redistribution aborts, the newest complete
// checkpoint is the previous natural write.
static void dmrDemoteCheckpointFields
(
    const fvMesh& mesh,
    DynamicList<regIOobject*>& demoted
)
{
    dmrDemoteFieldsOfType<volScalarField>(mesh, demoted);
    dmrDemoteFieldsOfType<volVectorField>(mesh, demoted);
    dmrDemoteFieldsOfType<volSphericalTensorField>(mesh, demoted);
    dmrDemoteFieldsOfType<volSymmTensorField>(mesh, demoted);
    dmrDemoteFieldsOfType<volTensorField>(mesh, demoted);
    dmrDemoteFieldsOfType<surfaceScalarField>(mesh, demoted);
    dmrDemoteFieldsOfType<surfaceVectorField>(mesh, demoted);
    dmrDemoteFieldsOfType<surfaceSphericalTensorField>(mesh, demoted);
    dmrDemoteFieldsOfType<surfaceSymmTensorField>(mesh, demoted);
    dmrDemoteFieldsOfType<surfaceTensorField>(mesh, demoted);
    dmrDemoteFieldsOfType<pointScalarField>(mesh, demoted);
    dmrDemoteFieldsOfType<pointVectorField>(mesh, demoted);
    dmrDemoteFieldsOfType<pointSphericalTensorField>(mesh, demoted);
    dmrDemoteFieldsOfType<pointSymmTensorField>(mesh, demoted);
    dmrDemoteFieldsOfType<pointTensorField>(mesh, demoted);
}


// Children currently writing an output time, one per region per write.
static std::vector<pid_t>& dmrWriterPids()
{
    static std::vector<pid_t> pids;
    return pids;
}


// Off with FOAM_DMR_ASYNC_WRITE=0, for MPI runtimes that misbehave when
// a process forks. Changes timing only, not what is written.
static bool dmrAsyncWriteEnabled()
{
    static const bool enabled = []()
    {
        const char* env = std::getenv("FOAM_DMR_ASYNC_WRITE");
        return !env || std::string(env) != "0";
    }();
    return enabled;
}


// Written last, after the data. An output time without it is incomplete
// or still being written by an asynchronous writer.
static void dmrWriteCompleteSentinel
(
    const Time& runTime,
    const std::string& timeName
)
{
    const fileName marker(runTime.path()/timeName/".complete");
    if (FILE* f = std::fopen(marker.c_str(), "w"))
    {
        std::fclose(f);
    }
}


// Reap the writers of the previous output time. Needed before the next
// write, a reconfiguration (which ends this process group) and the end
// of the run.
static void dmrWaitForWriters()
{
    std::vector<pid_t>& pids = dmrWriterPids();
    for (const pid_t pid : pids)
    {
        int status = 0;
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR)
        {}

        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        {
            WarningInFunction
                << "asynchronous write process " << label(pid)
                << " did not exit cleanly; the output time it was writing"
                << " has no .complete marker and should be discarded"
                << endl;
        }
    }
    pids.clear();
}


// Diagnostic: print the boundary a mesh presents to fvMeshDistribute.
// Off unless FOAM_DMR_DUMP_PATCHES is set.
static void dmrDumpPatches(const fvMesh& mesh, const char* what)
{
    static const bool on = (std::getenv("FOAM_DMR_DUMP_PATCHES") != nullptr);

    if (!on)
    {
        return;
    }

    const polyBoundaryMesh& pbm = mesh.poly().boundary();

    forAll(pbm, patchi)
    {
        Pout<< "DMRPATCH " << what << " region=" << mesh.name()
            << " nCells=" << mesh.nCells()
            << " patch=" << patchi
            << " name=" << pbm[patchi].name()
            << " type=" << pbm[patchi].type()
            << " size=" << pbm[patchi].size()
            << " isProc="
            << (isA<processorPolyPatch>(pbm[patchi]) ? 1 : 0)
            << endl;
    }
}


// Hand the cloned fields to the clone's registry. The clone outlives the
// lists dmrCloneFields fills, and a field deleted with its list checks
// itself out, so the clone would be written without fields.
template<class FieldType>
static void dmrStoreFields(PtrList<FieldType>& fields)
{
    forAll(fields, i)
    {
        regIOobject::store(fields.set(i, nullptr).ptr());
    }
}


// Clone one region of the live mesh with its checkpoint fields and
// redistribute the clone. Calling fvMeshDistribute on a clean mesh avoids
// the deadlock it hits on a mesh carrying live solver state.
//
// targetRanks = 0 (reconfiguration) uses numberOfSubdomains from the
// dictionary, i.e. the whole new world. A positive value (uniform write)
// sends everything to the first targetRanks ranks. Either way it is
// collective over the running world, which is wide enough because
// fvMeshDistribute needs max(source, target) ranks.
//
// Only the shrink and the uniform write use this, both of which still
// hold the live mesh. A grow has none (its processes are new) and reads
// through loadOrCreateMesh instead.
static autoPtr<fvMesh> dmrRedistributeToClone
(
    Time& runTime,
    const std::string& timeName,
    const word& regionName,
    const fvMesh& live,
    const label targetRanks
)
{
    autoPtr<fvMesh> meshPtr(dmrCloneLiveMesh(live, runTime, regionName));
    fvMesh& mesh = meshPtr();

    labelList finalDecomp;
    {
        dictionary decomposeDict
        (
            decompositionMethod::decomposeParDict(runTime)
        );

        if (targetRanks > 0)
        {
            decomposeDict.set("numberOfSubdomains", targetRanks);
        }

        autoPtr<decompositionMethod> distributor
        (
            decompositionMethod::NewDistributor(decomposeDict)
        );
        finalDecomp = dmrDecompose(distributor(), mesh);
    }

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

    dmrCloneFields(live, mesh, mesh, volScalars);
    dmrCloneFields(live, mesh, mesh, volVectors);
    dmrCloneFields(live, mesh, mesh, volSphereTensors);
    dmrCloneFields(live, mesh, mesh, volSymmTensors);
    dmrCloneFields(live, mesh, mesh, volTensors);
    dmrCloneFields(live, mesh, mesh, surfScalars);
    dmrCloneFields(live, mesh, mesh, surfVectors);
    dmrCloneFields(live, mesh, mesh, surfSphereTensors);
    dmrCloneFields(live, mesh, mesh, surfSymmTensors);
    dmrCloneFields(live, mesh, mesh, surfTensors);

    // Point fields (e.g. pointDisplacement), on the pointMesh. They must
    // be registered before distribute() to be transported.
    pointMesh& pMesh = const_cast<pointMesh&>(pointMesh::New(mesh));

    PtrList<pointScalarField>          pointScalars;
    PtrList<pointVectorField>          pointVectors;
    PtrList<pointSphericalTensorField> pointSphereTensors;
    PtrList<pointSymmTensorField>      pointSymmTensors;
    PtrList<pointTensorField>          pointTensors;

    dmrCloneFields(live, mesh, pMesh, pointScalars);
    dmrCloneFields(live, mesh, pMesh, pointVectors);
    dmrCloneFields(live, mesh, pMesh, pointSphereTensors);
    dmrCloneFields(live, mesh, pMesh, pointSymmTensors);
    dmrCloneFields(live, mesh, pMesh, pointTensors);

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

    dmrStoreFields(volScalars);
    dmrStoreFields(volVectors);
    dmrStoreFields(volSphereTensors);
    dmrStoreFields(volSymmTensors);
    dmrStoreFields(volTensors);
    dmrStoreFields(surfScalars);
    dmrStoreFields(surfVectors);
    dmrStoreFields(surfSphereTensors);
    dmrStoreFields(surfSymmTensors);
    dmrStoreFields(surfTensors);
    dmrStoreFields(pointScalars);
    dmrStoreFields(pointVectors);
    dmrStoreFields(pointSphereTensors);
    dmrStoreFields(pointSymmTensors);
    dmrStoreFields(pointTensors);

    const bool addedPatch = dmrAddTransientInternalPatch(mesh);

    // Drop the fvMesh geometry caches first. Sf, magSf, C and Cf are
    // slices over primitiveMesh arrays; the topology changes inside
    // distribute() free those arrays without clearing the slices, and on
    // a rank that gives away all its cells nothing else clears them
    // either (it adds no processor patch). The stale slices are then read
    // over freed memory: an intermittent SIGFPE on shrink.
    mesh.clearGeom();

    dmrDumpPatches(mesh, "region-before-distribute");

    fvMeshDistribute(mesh).distribute(finalDecomp);

    if (addedPatch)
    {
        dmrRemoveTransientInternalPatch(mesh);
    }

    mesh.setInstance(timeName);

    return meshPtr;
}


// Clone, redistribute and write one region (the uniform write). The
// shrink keeps all clones alive instead, to re-stitch couplings between
// regions before writing (dmrRestitchClones).
static void dmrRunFreshRedistribute
(
    Time& runTime,
    const std::string& timeName,
    const word& regionName,
    const fvMesh& live,
    const label targetRanks = 0
)
{
    autoPtr<fvMesh> meshPtr
    (
        dmrRedistributeToClone(runTime, timeName, regionName, live, targetRanks)
    );
    fvMesh& mesh = meshPtr();

    // Ranks outside the target layout hold no cells and write nothing
    if (targetRanks > 0 && Pstream::myProcNo() >= targetRanks)
    {
        return;
    }

    // The uniform write is done by a forked child so the time loop does
    // not wait for the disk. Not a thread: Pstream state is process-global
    // and two threads would mix message tags. Not a spawned MPI world: it
    // doubles the processes per node and exhausted the SysV segment limit
    // on full nodes. The child does no MPI, only local file I/O (the
    // uncollated write has no Pstream in it), and shares the parent's
    // pages copy-on-write. _exit, not exit: no atexit handlers, no stdio
    // flush, no MPI_Finalize.
    if (targetRanks > 0 && dmrAsyncWriteEnabled())
    {
        dmrWaitForWriters();

        const pid_t pid = ::fork();
        if (pid == 0)
        {
            mesh.write();
            dmrWriteCompleteSentinel(runTime, timeName);
            ::_exit(0);
        }
        else if (pid > 0)
        {
            dmrWriterPids().push_back(pid);
            return;
        }

        // fork failed: fall through and write here rather than lose the
        // output time
        WarningInFunction
            << "fork() for the asynchronous write failed ("
            << std::strerror(errno) << "); writing synchronously" << endl;
    }

    mesh.write();

    if (targetRanks > 0)
    {
        dmrWriteCompleteSentinel(runTime, timeName);
        return;
    }

    // constant/<region>/polyMesh/points is the motion reference shared
    // with the running solver. Rewrite it only when the solver is about
    // to be replaced (reconfiguration), never at a natural write. The
    // reference is in the time directory either way, as points0.
    if (targetRanks == 0)
    {
        dmrWritePoints0Constant(mesh);
    }
}


// Delete the fvMeshStitcher caches (fvMesh/polyFaces, which
// fvMeshDistribute cannot transport) from this rank's directory, at the
// given time and at constant. Without them the stitcher reconnects from
// geometry at startup. Run only after all regions are redistributed:
// regions read each other's caches through mapped patches. No-op
// without non-conformal couplings.
static void dmrSweepStitchCaches
(
    Time& runTime,
    const std::string& timeName,
    const wordList& regionNames,
    const bool keepTimeCache = false
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

        // The time-instance cache is kept only when it was just produced
        // for THIS layout by the destination re-stitch; the constant one
        // always belongs to the launch layout and is always stale
        if (!keepTimeCache && isDir(cacheT)) rmDir(cacheT);
        if (isDir(cacheC)) rmDir(cacheC);

        // tetBasePtIs is also a derivable cache, and fvMeshDistribute
        // does not remap it. A stale copy exists only on the ranks that
        // had data, which at restart then skip the collective
        // findFaceBasePts that the others enter. Absent, every rank
        // recomputes it together.
        const fileName meshDirT =
            runTime.path()/timeName/regionDir/"polyMesh";
        const fileName meshDirC =
            runTime.path()/runTime.constant()/regionDir/"polyMesh";

        rm(meshDirT/"tetBasePtIs");
        rm(meshDirC/"tetBasePtIs");
    }
}


// Swap meshes (fvMeshTopoChangers::meshToMesh). The solver reads
// constant/meshes/<meshTime>[/<region>]/polyMesh per rank at each swap
// time and adopts its decomposition, so after a reconfiguration the
// pieces must exist in the new world size.
//
// Slice one serial swap mesh (at the case root) into nNew pieces, in
// parallel. Collective over the current world.
static void dmrRedistributeSwapMesh
(
    Time& runTime,
    const word& meshTime,
    const word& regionName,     // word::null: default-region layout
    const label nNew
)
{
    const word name =
        regionName.empty() ? word(polyMesh::defaultRegion) : regionName;
    const fileName local = fileName("meshes")/meshTime;
    const fileName meshSubDir =
        regionName.empty()
      ? fileName(polyMesh::meshSubDir)
      : fileName(regionName)/polyMesh::meshSubDir;
    const fileName pieceDir = runTime.path()/runTime.constant()/local/meshSubDir;

    // Slice the serial mesh, which never changes, rather than re-arrange
    // the existing pieces: re-arranging has to reconcile the old layout's
    // couplings, and the engine's sliding interfaces do not survive it.
    //
    // This is redistributePar's documented use as a parallel
    // decomposePar: the undecomposed mesh on one rank, the others empty.
    // The mesh has to be copied into the master's processor directory,
    // since the file handler falls back to the case root only for global
    // files, and mesh files are not global.
    const fileName serialDir =
        fileName(runTime.globalPath())/runTime.constant()/local;
    const fileName localDir = runTime.path()/runTime.constant()/local;

    bool haveSerial = false;
    if (Pstream::master())
    {
        haveSerial = isFile(serialDir/meshSubDir/"faces");

        if (haveSerial)
        {
            // Stale pieces out, the serial original in
            if (isDir(localDir)) rmDir(localDir);
            mkDir(localDir);
            cp(serialDir/meshSubDir.path(), localDir);
        }
    }
    Pstream::scatter(haveSerial);

    if (!haveSerial)
    {
        FatalErrorInFunction
            << "no serial swap mesh under " << runTime.constant()/local
            << " for " << meshTime
            << (regionName.empty() ? "" : " of region " + regionName)
            << exit(FatalError);
    }

    // Everyone else starts empty
    if (!Pstream::master() && isDir(localDir))
    {
        rmDir(localDir);
    }

    boolList haveMesh(Pstream::nProcs(), false);
    haveMesh[Pstream::masterNo()] = true;

    // The master's boundary entries, for the empty meshes (the same
    // reading trick as loadOrCreateMesh: the file's class is not a list)
    PtrList<entry> patchEntries;
    if (Pstream::master())
    {
        const word oldTypeName = IOPtrList<entry>::typeName;
        const_cast<word&>(IOPtrList<entry>::typeName) = word::null;
        IOPtrList<entry> dictList
        (
            IOobject
            (
                "boundary",
                runTime.constant(),
                local/meshSubDir,
                runTime,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false
            )
        );
        const_cast<word&>(IOPtrList<entry>::typeName) = oldTypeName;
        const_cast<word&>(dictList.type()) = dictList.headerClassName();
        patchEntries.transfer(dictList);

        for
        (
            int slave = Pstream::firstSlave();
            slave <= Pstream::lastSlave();
            slave++
        )
        {
            OPstream toSlave(Pstream::commsTypes::scheduled, slave);
            toSlave << patchEntries;
        }
    }
    else
    {
        IPstream fromMaster(Pstream::commsTypes::scheduled, Pstream::masterNo());
        fromMaster >> patchEntries;
    }


    // The master reads the serial mesh; the mesh that gets distributed is
    // rebuilt from its primitives on every rank, as in dmrCloneLiveMesh,
    // and the zones are copied over further down.
    //
    // Only the master reads, so parallel communication is switched off
    // until every rank reaches the collective addFvPatches, as
    // loadOrCreateMesh does. Otherwise the empty ranks' messages cross
    // the master's read.
    const bool oldParRun = Pstream::parRun();
    Pstream::parRun() = false;

    autoPtr<fvMesh> readPtr;

    if (haveMesh[Pstream::myProcNo()])
    {
        readPtr.reset
        (
            new fvMesh
            (
                IOobject
                (
                    name,
                    runTime.constant(),
                    local,
                    runTime,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE,
                    false
                ),
                false
            )
        );
    }

    pointField points;
    faceList faces;
    labelList allOwner;
    labelList allNeighbour;

    if (readPtr.valid())
    {
        points = readPtr().points();
        faces = readPtr().faces();
        allOwner = readPtr().faceOwner();
        allNeighbour = readPtr().faceNeighbour();
    }

    autoPtr<fvMesh> meshPtr
    (
        new fvMesh
        (
            IOobject
            (
                name,
                runTime.constant(),
                local,
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

    {
        // Every rank builds its patches fresh from the boundary entries,
        // the master keeping its face counts. The master must not clone
        // the patches of the mesh it read: that serial read already
        // computed their geometry, and some patches cache it and then skip
        // the calculation, collectives included (wedgePolyPatch with its
        // axis set). Cloned on the master and fresh elsewhere, the ranks
        // fall out of step inside addFvPatches (MPI_ERR_TRUNCATE).
        List<polyPatch*> patches(patchEntries.size());
        label nPatches = 0;

        forAll(patchEntries, patchi)
        {
            const entry& e = patchEntries[patchi];

            dictionary patchDict(e.dict());
            if (!readPtr.valid())
            {
                patchDict.set("nFaces", 0);
                patchDict.set("startFace", 0);
            }

            autoPtr<polyPatch> pp
            (
                polyPatch::New
                (
                    e.keyword(),
                    patchDict,
                    nPatches,
                    meshPtr().poly().boundary()
                )
            );

            // Processor patches of any kind are decided by class, so the
            // non-conformal processor cyclics go too. A serial mesh has
            // none; if the master finds one, the input is not serial and
            // dropping it would lose faces.
            if (isA<processorPolyPatch>(pp()))
            {
                if (readPtr.valid())
                {
                    FatalErrorInFunction
                        << "swap mesh " << meshTime << " under "
                        << runTime.constant()/local
                        << " has processor patch " << e.keyword()
                        << "; expected an undecomposed mesh"
                        << exit(FatalError);
                }
                continue;
            }

            patches[nPatches++] = pp.ptr();
        }
        patches.setSize(nPatches);

        Pstream::parRun() = oldParRun;

        meshPtr().addFvPatches(patches, true);
    }

    fvMesh& mesh = meshPtr();


    // A mesh built from primitives has no zones, so the master's are
    // copied from the mesh it read (the engine's swap meshes carry face
    // zones the solver needs after a swap).
    if (readPtr.valid())
    {
        const pointZoneList& rpz = readPtr().pointZones();
        const faceZoneList& rfz = readPtr().faceZones();
        const cellZoneList& rcz = readPtr().cellZones();

        List<pointZone*> pz(rpz.size());
        forAll(rpz, i)
        {
            pz[i] = new pointZone(rpz[i].name(), rpz[i], mesh.pointZones());
        }
        List<faceZone*> fz(rfz.size());
        forAll(rfz, i)
        {
            fz[i] = new faceZone
            (
                rfz[i].name(),
                rfz[i],
                rfz[i].flipMap(),
                mesh.faceZones()
            );
        }
        List<cellZone*> cz(rcz.size());
        forAll(rcz, i)
        {
            cz[i] = new cellZone(rcz[i].name(), rcz[i], mesh.cellZones());
        }

        mesh.addZones(pz, fz, cz);
    }

    readPtr.clear();

    // Empty zones of the same names on the ranks without a mesh, so
    // distribute sees the same zones everywhere
    wordList pointZoneNames(mesh.pointZones().toc());
    Pstream::scatter(pointZoneNames);
    wordList faceZoneNames(mesh.faceZones().toc());
    Pstream::scatter(faceZoneNames);
    wordList cellZoneNames(mesh.cellZones().toc());
    Pstream::scatter(cellZoneNames);

    if (!haveMesh[Pstream::myProcNo()])
    {
        List<pointZone*> pz(pointZoneNames.size());
        forAll(pointZoneNames, i)
        {
            pz[i] = new pointZone
            (
                pointZoneNames[i],
                labelList(0),
                mesh.pointZones()
            );
        }
        List<faceZone*> fz(faceZoneNames.size());
        forAll(faceZoneNames, i)
        {
            fz[i] = new faceZone
            (
                faceZoneNames[i],
                labelList(0),
                boolList(0),
                mesh.faceZones()
            );
        }
        List<cellZone*> cz(cellZoneNames.size());
        forAll(cellZoneNames, i)
        {
            cz[i] = new cellZone
            (
                cellZoneNames[i],
                labelList(0),
                mesh.cellZones()
            );
        }
        mesh.addZones(pz, fz, cz);
    }

    autoPtr<decompositionMethod> distributor
    (
        decompositionMethod::NewDistributor
        (
            decompositionMethod::decomposeParDict(runTime)
        )
    );

    // Geometric decomposition, as redistributePar uses when all cells
    // start on one rank (the weights overload leaves them all there)
    const labelList finalDecomp = dmrDecompose(distributor(), mesh);

    // Needed even without fields: otherwise exposed faces are parked on
    // the first non-empty patch, a coupled one on the engine, and the
    // coupled-face reordering fails
    const bool addedPatch = dmrAddTransientInternalPatch(mesh);

    // No stale geometry slices into fvMeshDistribute (see
    // dmrRedistributeToClone)
    mesh.clearGeom();

    fvMeshDistribute(mesh).distribute(finalDecomp);

    if (addedPatch)
    {
        dmrRemoveTransientInternalPatch(mesh);
    }

    // Ranks beyond the new world are left empty; the caller removes them
    if (Pstream::myProcNo() < nNew)
    {
        // Distribution leaves the instance at the current time; the
        // pieces belong in constant/meshes/<t>
        mesh.setInstance(runTime.constant());

        mesh.write();

        // Left over from the serial copy on the master
        rm(pieceDir/"cellProc");

        // Left over from the previous decomposition, which the write does
        // not overwrite
        if
        (
            mesh.pointZones().empty()
         && mesh.faceZones().empty()
         && mesh.cellZones().empty()
        )
        {
            rm(pieceDir/"pointZones");
            rm(pieceDir/"faceZones");
            rm(pieceDir/"cellZones");
        }
        rm(pieceDir/"cellProcAddressing");
        rm(pieceDir/"faceProcAddressing");
        rm(pieceDir/"pointProcAddressing");
        rm(pieceDir/"boundaryProcAddressing");
    }
}


static void dmrRedistributeSwapMeshes(Time& runTime, const label nNew)
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

    const double t0 = MPI_Wtime();

    forAll(meshTimes, meshi)
    {
        dmrRedistributeSwapMesh
        (
            runTime, meshTimes[meshi], meshRegions[meshi], nNew
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        char elapsed[32];
        std::snprintf(elapsed, sizeof(elapsed), "%.3f", MPI_Wtime() - t0);

        Info<< "DMR: redistributed " << meshTimes.size()
            << " swap mesh(es) under constant/meshes to " << nNew
            << " ranks in " << elapsed << " s" << endl;
    }
}


// * * * * * * * * * Uniform (MIN-layout) output write  * * * * * * * * * * //
//
// A malleable run would otherwise write each output time in whatever
// layout it is running at, and a decomposed case must have one
// decomposition across all times. So every natural write is emitted in
// the MIN layout (policy-min nodes x procs-per-node), which the run can
// always produce: a clone of each live mesh is redistributed to MIN and
// written by a forked child (dmrUniformWriteInPlace).
//
// dmrSpawnUniformWrite and dmrRelayoutFromParent are an earlier version
// that did this in a spawned MPI world. They are retained but not
// called: on full nodes the extra processes exhausted the SysV segment
// limit.

// Target world size for the uniform output: policy-min nodes times
// procs-per-node, from the environment the launcher configures DMR with
// (DMR has no getter for the policy minimum).
static int dmrMinWorldSize()
{
    const char* minNodes = std::getenv("DMR_DEFAULT_POLICY_MIN");
    const char* ppn = std::getenv("DMR_PROCS_PER_NODE");

    const int n = minNodes ? std::atoi(minNodes) : 1;
    const int p = ppn ? std::atoi(ppn) : 1;

    return (n > 0 ? n : 1) * (p > 0 ? p : 1);
}


// Emit every natural write in the MIN layout, so the output series has
// one decomposition. On by default; FOAM_DMR_UNIFORM_WRITE=0 writes in
// the running layout instead, which leaves the series unreadable across
// reconfigurations.
static bool dmrUniformWriteEnabled()
{
    static const bool enabled = []()
    {
        const char* env = std::getenv("FOAM_DMR_UNIFORM_WRITE");
        return !env || std::string(env) != "0";
    }();
    return enabled;
}


// Whether superseded DMR-forced checkpoints are deleted as the run
// proceeds. On by default; FOAM_DMR_KEEP_CHECKPOINTS=1 keeps them all,
// for debugging a restart. Independent of the uniform write: only the
// newest checkpoint is ever read, and keeping the rest costs a full mesh
// and fields per reconfiguration, enough to fill a filesystem.
static bool dmrPurgeCheckpointsEnabled()
{
    static const bool enabled = []()
    {
        const char* env = std::getenv("FOAM_DMR_KEEP_CHECKPOINTS");
        return !env || std::string(env) == "0";
    }();
    return enabled;
}


// Spawned-helper uniform write: retained, not called (see the section
// header). The helper world still writing, or MPI_COMM_NULL.
static MPI_Comm dmrHelperComm = MPI_COMM_NULL;


// Wait for the outstanding helper, if any. MPI_Comm_disconnect returns
// once both worlds have called it, and the helper calls it last, so this
// blocks until the helper has finished writing. Called only where a
// running helper would otherwise be killed: before spawning another, at
// a reconfiguration and at the end of the run.
static void dmrWaitForHelper()
{
    if (dmrHelperComm == MPI_COMM_NULL)
    {
        return;
    }

    // Return errors rather than abort the solver if the helper died. The
    // communicator is destroyed below, so the handler is not restored.
    MPI_Comm_set_errhandler(dmrHelperComm, MPI_ERRORS_RETURN);

    const int rc = MPI_Comm_disconnect(&dmrHelperComm);

    if (rc != MPI_SUCCESS)
    {
        WarningInFunction
            << "MPI_Comm_disconnect from the uniform-write helper failed"
            << " (code " << rc << "). The output time it was writing may"
            << " be incomplete; check for its .complete sentinel." << endl;

        // Drop the handle rather than retry and block on a dead peer
        dmrHelperComm = MPI_COMM_NULL;
    }
}


// Scope guard so no exit path of dmrCheckpoint misses the wait
namespace
{
struct dmrHelperBarrier
{
    ~dmrHelperBarrier() { dmrWaitForHelper(); }
};
}


// Send/receive one serialised blob over the spawn intercommunicator.
// Parent rank i talks only to child rank i.
static void dmrSendBlob(const std::string& s, MPI_Comm inter, int peer)
{
    int n = static_cast<int>(s.size());
    MPI_Send(&n, 1, MPI_INT, peer, 0, inter);
    MPI_Send(s.data(), n, MPI_CHAR, peer, 1, inter);
}


static std::string dmrRecvBlob(MPI_Comm inter, int peer)
{
    int n = 0;
    MPI_Recv(&n, 1, MPI_INT, peer, 0, inter, MPI_STATUS_IGNORE);
    std::string s(n, '\0');
    MPI_Recv(&s[0], n, MPI_CHAR, peer, 1, inter, MPI_STATUS_IGNORE);
    return s;
}


// Serialise this rank's fields of one type, a sub-dictionary per field.
// AUTO_WRITE only, as in dmrCloneFields.
template<class GeoField>
static void dmrWriteFieldsBlob(const fvMesh& mesh, OStringStream& os)
{
    const HashTable<const GeoField*> fields(mesh.lookupClass<GeoField>());
    const wordList names(fields.sortedToc());

    os  << GeoField::typeName << nl << token::BEGIN_BLOCK << nl;

    forAll(names, i)
    {
        const GeoField& f = *fields[names[i]];

        if (f.writeOpt() != IOobject::AUTO_WRITE) continue;

        os  << names[i] << nl << token::BEGIN_BLOCK << nl;
        f.writeData(os);
        os  << nl << token::END_BLOCK << nl;
    }

    os  << token::END_BLOCK << nl;
}


// Rebuild one field type on the helper's mesh from the received blob.
template<class GeoField>
static void dmrReadFieldsBlob
(
    const dictionary& all,
    const fvMesh& mesh,
    const typename GeoField::Mesh& fieldMesh,
    PtrList<GeoField>& fields
)
{
    if (!all.found(GeoField::typeName))
    {
        return;
    }

    const dictionary& sub = all.subDict(GeoField::typeName);
    const wordList names(sub.toc());

    fields.setSize(names.size());
    forAll(names, i)
    {
        fields.set
        (
            i,
            new GeoField
            (
                IOobject
                (
                    names[i],
                    mesh.time().name(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                fieldMesh,
                sub.subDict(names[i])
            )
        );
    }
}


void dmrSpawnUniformWrite(Time& runTime, bool allRegions)
{
    if (!dmrUniformWriteEnabled())
    {
        return;
    }

    // At most one helper at a time
    dmrWaitForHelper();

    const int minSize = dmrMinWorldSize();
    int worldSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Already uniform: the natural write is in the target layout
    if (worldSize <= minSize)
    {
        return;
    }

    const std::string casePath = runTime.globalPath();

    // Region set, in a deterministic order on every rank
    wordList regionNames;
    {
        HashTable<fvMesh*> liveRegions(runTime.lookupClass<fvMesh>());
        regionNames = liveRegions.sortedToc();
    }

    if (regionNames.empty())
    {
        return;
    }

    // Spawn a helper world of the same size. argv mirrors how the
    // solver itself was launched so the helper resolves the same case.
    const std::string helper
    (
        std::string(std::getenv("FOAM_APPBIN") ? std::getenv("FOAM_APPBIN") : ".")
      + "/dmrRelayout"
    );

    std::vector<char*> argvChild;
    static char optCase[] = "-case";
    static char optParallel[] = "-parallel";
    std::vector<char> caseBuf(casePath.begin(), casePath.end());
    caseBuf.push_back('\0');

    argvChild.push_back(optCase);
    argvChild.push_back(caseBuf.data());
    argvChild.push_back(optParallel);
    argvChild.push_back(nullptr);

    std::vector<char> helperBuf(helper.begin(), helper.end());
    helperBuf.push_back('\0');

    // Provision the helper as DMR provisions its own spawned processes
    // (dmr_core.c): the PMIX_* keys add hosts and slots for the new
    // processes and allow overloading. With MPI_INFO_NULL the spawn fails
    // ("all nodes already filled") when the solver fills the allocation.
    MPI_Info spawnInfo = MPI_INFO_NULL;
    {
        char procName[MPI_MAX_PROCESSOR_NAME];
        int nameLen = 0;
        MPI_Get_processor_name(procName, &nameLen);

        std::vector<char> allNames
        (
            std::size_t(worldSize)*MPI_MAX_PROCESSOR_NAME, '\0'
        );
        MPI_Allgather
        (
            procName, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
            allNames.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
            MPI_COMM_WORLD
        );

        // One entry per distinct node, carrying the number of helper
        // ranks to place there (mirrors this world's own distribution)
        std::vector<std::string> nodes;
        std::vector<int> counts;
        for (int r = 0; r < worldSize; ++r)
        {
            const std::string n(&allNames[std::size_t(r)*MPI_MAX_PROCESSOR_NAME]);
            const auto it = std::find(nodes.begin(), nodes.end(), n);
            if (it == nodes.end())
            {
                nodes.push_back(n);
                counts.push_back(1);
            }
            else
            {
                counts[it - nodes.begin()]++;
            }
        }

        std::string hostSpec;
        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            if (i) hostSpec += ",";
            hostSpec += nodes[i] + ":" + std::to_string(counts[i]);
        }

        MPI_Info_create(&spawnInfo);
        MPI_Info_set(spawnInfo, "PMIX_ADD_HOST", hostSpec.c_str());
        MPI_Info_set(spawnInfo, "PMIX_HOST", hostSpec.c_str());
        MPI_Info_set(spawnInfo, "PMIX_MAPBY", ":OVERSUBSCRIBE");
        MPI_Info_set(spawnInfo, "PMIX_BINDTO", ":overload-allowed");
    }

    MPI_Comm inter = MPI_COMM_NULL;
    std::vector<int> errcodes(worldSize, 0);

    // A refused spawn leaves this output time in the running layout; it
    // must not abort the solver
    MPI_Errhandler oldHandler = MPI_ERRHANDLER_NULL;
    MPI_Comm_get_errhandler(MPI_COMM_WORLD, &oldHandler);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    const int rc = MPI_Comm_spawn
    (
        helperBuf.data(),
        argvChild.data(),
        worldSize,
        spawnInfo,
        0,
        MPI_COMM_WORLD,
        &inter,
        errcodes.data()
    );

    MPI_Comm_set_errhandler(MPI_COMM_WORLD, oldHandler);

    if (spawnInfo != MPI_INFO_NULL)
    {
        MPI_Info_free(&spawnInfo);
    }

    if (rc != MPI_SUCCESS || inter == MPI_COMM_NULL)
    {
        if (Pstream::master())
        {
            dmrLifecycleLog
            (
                casePath,
                "Uniform write: MPI_Comm_spawn failed; natural write left"
                " in the running layout."
            );
        }
        return;
    }

    const int myRank = Pstream::myProcNo();

    // Header: what the helper needs before it can interpret the rest
    {
        // Time name quoted so "0.05" reads back as a word. 17 digits so
        // deltaT round-trips exactly.
        OStringStream os;
        os.precision(17);
        os  << "time" << token::SPACE << string(runTime.name())
            << token::END_STATEMENT << nl
            << "timeIndex" << token::SPACE << runTime.timeIndex()
            << token::END_STATEMENT << nl
            << "deltaT" << token::SPACE << runTime.deltaTValue()
            << token::END_STATEMENT << nl
            << "minRanks" << token::SPACE << minSize
            << token::END_STATEMENT << nl
            << "regions" << token::SPACE << regionNames
            << token::END_STATEMENT << nl;

        dmrSendBlob(os.str(), inter, myRank);
    }

    forAll(regionNames, regioni)
    {
        const fvMesh& mesh =
            runTime.lookupObject<fvMesh>(regionNames[regioni]);

        // Mesh primitives, in the order fvMeshDistribute::sendMesh uses
        {
            OStringStream os;
            os  << mesh.points()
                << CompactListList<label>(mesh.faces())
                << mesh.faceOwner()
                << mesh.faceNeighbour()
                << mesh.poly().boundary();

            dmrSendBlob(os.str(), inter, myRank);
        }

        // Fields, as one dictionary of per-type sub-dictionaries
        {
            OStringStream os;
            os.precision(17);   // exact round-trip of an IEEE double

            dmrWriteFieldsBlob<volScalarField>(mesh, os);
            dmrWriteFieldsBlob<volVectorField>(mesh, os);
            dmrWriteFieldsBlob<volSphericalTensorField>(mesh, os);
            dmrWriteFieldsBlob<volSymmTensorField>(mesh, os);
            dmrWriteFieldsBlob<volTensorField>(mesh, os);
            dmrWriteFieldsBlob<surfaceScalarField>(mesh, os);
            dmrWriteFieldsBlob<surfaceVectorField>(mesh, os);
            dmrWriteFieldsBlob<surfaceSphericalTensorField>(mesh, os);
            dmrWriteFieldsBlob<surfaceSymmTensorField>(mesh, os);
            dmrWriteFieldsBlob<surfaceTensorField>(mesh, os);
            dmrWriteFieldsBlob<pointScalarField>(mesh, os);
            dmrWriteFieldsBlob<pointVectorField>(mesh, os);
            dmrWriteFieldsBlob<pointSphericalTensorField>(mesh, os);
            dmrWriteFieldsBlob<pointSymmTensorField>(mesh, os);
            dmrWriteFieldsBlob<pointTensorField>(mesh, os);

            dmrSendBlob(os.str(), inter, myRank);
        }
    }

    // Not disconnected here, which would block until the helper finished;
    // dmrWaitForHelper does that later
    dmrHelperComm = inter;

    if (Pstream::master())
    {
        dmrLifecycleLog
        (
            casePath,
            "Uniform write at t=" + runTime.name() + ": handed "
          + std::to_string(regionNames.size()) + " region(s) to a "
          + std::to_string(worldSize) + "-rank helper for "
          + std::to_string(minSize) + "-way output."
        );
    }
}


// A DMR-forced checkpoint is restart state, not output. Mark each with a
// sentinel and delete all but the newest (the restart state) from this
// rank's directory. Times are ordered as numbers, not strings.
static void dmrPurgeSupersededCheckpoints
(
    const Time& runTime,
    const std::string& timeName
)
{
    const std::string dir(runTime.path()/fileName(timeName));

    if (FILE* s = std::fopen((dir + "/.dmr_checkpoint").c_str(), "w"))
    {
        std::fclose(s);
    }

    const std::string pattern(std::string(runTime.path()) + "/*/.dmr_checkpoint");

    glob_t g;
    if (glob(pattern.c_str(), 0, nullptr, &g) != 0)
    {
        return;
    }

    const std::string suffix("/.dmr_checkpoint");
    std::vector<std::pair<double, std::string>> stamped;
    stamped.reserve(g.gl_pathc);

    for (size_t i = 0; i < g.gl_pathc; ++i)
    {
        std::string p(g.gl_pathv[i]);
        p.resize(p.size() - suffix.size());

        const std::string tName(p.substr(p.find_last_of('/') + 1));
        try
        {
            stamped.emplace_back(std::stod(tName), p);
        }
        catch (const std::exception&)
        {}
    }
    globfree(&g);

    std::sort(stamped.begin(), stamped.end());

    // Keep the newest; the rest are superseded restart state
    for (size_t i = 0; i + 1 < stamped.size(); ++i)
    {
        Foam::rmDir(stamped[i].second);
    }
}


void dmrUniformWriteInPlace(Time& runTime, bool allRegions)
{
    // Every live region is written regardless: the series is only
    // readable if all of them are in the MIN layout
    (void)allRegions;

    const label minRanks = dmrMinWorldSize();
    const std::string timeName = runTime.name();

    // Same order on every rank: everything below is collective
    wordList regionNames;
    {
        HashTable<fvMesh*> liveRegions(runTime.lookupClass<fvMesh>());
        regionNames = liveRegions.sortedToc();
    }

    // Fall back to an ordinary write before the first collective; past
    // it, a failure is not recoverable
    if (minRanks < 1 || regionNames.empty())
    {
        runTime.write();
        return;
    }

    // Redistribute a clone of each live mesh to the MIN layout. The
    // solver carries on at its own width, so the live mesh is never
    // touched.
    forAll(regionNames, i)
    {
        dmrRunFreshRedistribute
        (
            runTime,
            timeName,
            regionNames[i],
            runTime.lookupObject<fvMesh>(regionNames[i]),
            minRanks
        );
    }

    // uniform/time, which the clones do not carry. Not runTime.write():
    // that would also write the live meshes in the running layout over
    // the MIN layout just written.
    runTime.writeTimeDict();

    // Derivable caches, as after any redistribute
    dmrSweepStitchCaches(runTime, timeName, regionNames);

    // Ranks outside the MIN layout write nothing; remove any directory a
    // previous generation left for this time
    if (Pstream::myProcNo() >= minRanks)
    {
        const fileName stale(runTime.path()/timeName);
        if (isDir(stale))
        {
            rmDir(stale);
        }
    }

    // Not for file ordering: it keeps the ranks' skew from the file work
    // above out of the next timestep, where TALP would charge it to the
    // solver and lower the CE the controller reconfigures on
    MPI_Barrier(MPI_COMM_WORLD);
}


void dmrWrite(Time& runTime, bool allRegions)
{
    int worldSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const bool uniform =
        dmrUniformWriteEnabled()
     && runTime.writeTime()
     && worldSize > dmrMinWorldSize();

    if (uniform)
    {
        dmrUniformWriteInPlace(runTime, allRegions);
        return;
    }

    // At MIN width the running layout is the target: an ordinary write,
    // but with its own mesh. Otherwise a reader takes the nearest earlier
    // mesh, which may be a different MIN-wide decomposition from a
    // uniform write, and reads the fields against the wrong partition.
    // setInstance re-arms points and topology for writing to this time
    // without touching the mesh itself.
    if (dmrUniformWriteEnabled() && runTime.writeTime())
    {
        HashTable<fvMesh*> liveRegions(runTime.lookupClass<fvMesh>());
        forAllIter(HashTable<fvMesh*>, liveRegions, iter)
        {
            iter()->setInstance(runTime.name());
        }
    }

    runTime.write();

    // Same marker the asynchronous writer drops, so every output time can
    // be checked the same way
    if (dmrUniformWriteEnabled() && runTime.writeTime())
    {
        dmrWriteCompleteSentinel(runTime, runTime.name());
    }
}


void dmrRelayoutFromParent(Time& runTime)
{
    MPI_Comm parent = MPI_COMM_NULL;
    MPI_Comm_get_parent(&parent);

    if (parent == MPI_COMM_NULL)
    {
        FatalErrorInFunction
            << "dmrRelayout must be started by MPI_Comm_spawn from a"
            << " running solver; it has no parent communicator."
            << exit(FatalError);
    }

    const int myRank = Pstream::myProcNo();

    // Header
    word timeName;
    label minRanks = 1;
    label timeIndex = 0;
    scalar deltaT = 0;
    wordList regionNames;
    {
        IStringStream is(dmrRecvBlob(parent, myRank));
        const dictionary header(is);

        string timeStr;
        header.lookup("time") >> timeStr;
        timeName = word(timeStr);

        minRanks = readLabel(header.lookup("minRanks"));
        timeIndex = readLabel(header.lookup("timeIndex"));
        deltaT = readScalar(header.lookup("deltaT"));
        header.lookup("regions") >> regionNames;
    }

    // Adopt the parent's position in time rather than this process's own
    // start state. uniform/time carries the time index and deltaT a
    // restart resumes from, and with an adjustable time step those are
    // the SOLVER's current values, which the helper cannot derive.
    runTime.setTime(instant(timeName), timeIndex);
    runTime.setDeltaT(deltaT);

    // Receive everything first. The disconnect comes last: the parent
    // uses it as the "finished writing" signal (dmrWaitForHelper).
    List<string> meshBlobs(regionNames.size());
    List<string> fieldBlobs(regionNames.size());

    forAll(regionNames, regioni)
    {
        meshBlobs[regioni] = dmrRecvBlob(parent, myRank);
        fieldBlobs[regioni] = dmrRecvBlob(parent, myRank);
    }

    forAll(regionNames, regioni)
    {
        const word& regionName = regionNames[regioni];

        // Mesh, from the primitives the parent sent
        autoPtr<fvMesh> meshPtr;
        {
            IStringStream is(meshBlobs[regioni]);

            pointField points(is);
            faceList faces = CompactListList<label>(is).list<face>();
            labelList allOwner(is);
            labelList allNeighbour(is);
            PtrList<entry> patchEntries(is);

            meshPtr.reset
            (
                new fvMesh
                (
                    IOobject
                    (
                        regionName,
                        timeName,
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

            List<polyPatch*> patches(patchEntries.size());
            forAll(patchEntries, patchi)
            {
                patches[patchi] = polyPatch::New
                (
                    patchEntries[patchi].keyword(),
                    patchEntries[patchi].dict(),
                    patchi,
                    meshPtr().poly().boundary()
                ).ptr();
            }
            meshPtr().addFvPatches(patches, false);
        }

        fvMesh& mesh = meshPtr();

        // Fields
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
        PtrList<pointScalarField>            pointScalars;
        PtrList<pointVectorField>            pointVectors;
        PtrList<pointSphericalTensorField>   pointSphereTensors;
        PtrList<pointSymmTensorField>        pointSymmTensors;
        PtrList<pointTensorField>            pointTensors;

        {
            IStringStream is(fieldBlobs[regioni]);
            const dictionary all(is);

            pointMesh& pMesh = const_cast<pointMesh&>(pointMesh::New(mesh));

            dmrReadFieldsBlob(all, mesh, mesh, volScalars);
            dmrReadFieldsBlob(all, mesh, mesh, volVectors);
            dmrReadFieldsBlob(all, mesh, mesh, volSphereTensors);
            dmrReadFieldsBlob(all, mesh, mesh, volSymmTensors);
            dmrReadFieldsBlob(all, mesh, mesh, volTensors);
            dmrReadFieldsBlob(all, mesh, mesh, surfScalars);
            dmrReadFieldsBlob(all, mesh, mesh, surfVectors);
            dmrReadFieldsBlob(all, mesh, mesh, surfSphereTensors);
            dmrReadFieldsBlob(all, mesh, mesh, surfSymmTensors);
            dmrReadFieldsBlob(all, mesh, mesh, surfTensors);
            dmrReadFieldsBlob(all, mesh, pMesh, pointScalars);
            dmrReadFieldsBlob(all, mesh, pMesh, pointVectors);
            dmrReadFieldsBlob(all, mesh, pMesh, pointSphereTensors);
            dmrReadFieldsBlob(all, mesh, pMesh, pointSymmTensors);
            dmrReadFieldsBlob(all, mesh, pMesh, pointTensors);
        }

        // Redistribute this world's N-way copy down to the MIN layout.
        // Ranks at or beyond minRanks end up empty and are cleaned up
        // after the write.
        labelList finalDecomp;
        {
            dictionary decomposeDict
            (
                decompositionMethod::decomposeParDict(runTime)
            );
            decomposeDict.set("numberOfSubdomains", minRanks);

            autoPtr<decompositionMethod> distributor
            (
                decompositionMethod::NewDistributor(decomposeDict)
            );
            finalDecomp = dmrDecompose(distributor(), mesh);
        }

        const bool addedPatch = dmrAddTransientInternalPatch(mesh);

        fvMeshDistribute(mesh).distribute(finalDecomp);

        if (addedPatch)
        {
            dmrRemoveTransientInternalPatch(mesh);
        }

        mesh.setInstance(timeName);
        mesh.write();
    }

    // The solver performs no I/O at a natural write any more, so the
    // non-field state it used to emit (uniform/time above all) is
    // written here instead. The meshes built above are non-registered,
    // so this writes only what Time itself owns.
    runTime.writeNow();

    // Derivable caches, as after any redistribute
    dmrSweepStitchCaches(runTime, timeName, regionNames);

    MPI_Barrier(MPI_COMM_WORLD);

    // Ranks outside the MIN layout hold nothing at this time: drop the
    // empty directories they just wrote so the output series is exactly
    // minRanks wide.
    if (Pstream::myProcNo() >= minRanks)
    {
        const fileName stale(runTime.path()/timeName);
        if (isDir(stale))
        {
            rmDir(stale);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Sentinel, written LAST by every rank that keeps this time: an
    // output directory without it was produced by a helper that died
    // before finishing, so the time is suspect and should be discarded
    // rather than trusted. The parent's wait normally prevents that, but
    // it cannot rescue a helper that crashes outright.
    if (Pstream::myProcNo() < minRanks)
    {
        const fileName marker(runTime.path()/timeName/".complete");
        if (FILE* f = std::fopen(marker.c_str(), "w"))
        {
            std::fclose(f);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Signals the parent that this helper is done (its dmrWaitForHelper
    // returns here). Must be the last thing this process does.
    MPI_Comm_disconnect(&parent);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Shared with the shrink path, defined with it below
static bool dmrHasNonConformalPatches(const PtrList<fvMesh>& clones);
static void dmrPostConstructAll(PtrList<fvMesh>& meshes);
static void dmrConnectAll(const wordList& regionNames, PtrList<fvMesh>& meshes);
static void dmrConformAll(const wordList& regionNames, PtrList<fvMesh>& meshes);
static void dmrFillAfterStitch
(
    fvMesh& mesh,
    const word& regionName,
    const std::string& timeName,
    const label nNew
);
static void dmrLoadSolverLibraries(const Time& runTime);


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

    dmrLoadSolverLibraries(runTime);

    const label nRegions = regionNames.size();

    // As in the shrink, every region goes through each phase before the
    // next: couplings between regions need all of them at once.
    //
    // Phase 1: every region's mesh, loaded (or created empty on the new
    // ranks) and registered under its name, so mapped walls find their
    // neighbours when stitched.
    PtrList<fvMesh> meshes(nRegions);
    List<fileName> meshSubDirs(nRegions);
    List<fileName> masterInstDirs(nRegions);
    List<boolList> haveMeshes(nRegions);

    forAll(regionNames, regioni)
    {
        const word& regionName = regionNames[regioni];

        fileName& meshSubDir = meshSubDirs[regioni];
        if (regionName == polyMesh::defaultRegion)
        {
            meshSubDir = polyMesh::meshSubDir;
        }
        else
        {
            meshSubDir = regionName / polyMesh::meshSubDir;
        }

        // Only the master is guaranteed to have the mesh on disk
        fileName& masterInstDir = masterInstDirs[regioni];
        if (Pstream::master())
        {
            masterInstDir = runTime.findInstance(meshSubDir, "points");
        }
        Pstream::scatter(masterInstDir);

        boolList& haveMesh = haveMeshes[regioni];
        haveMesh.setSize(Pstream::nProcs(), false);
        haveMesh[Pstream::myProcNo()] =
            isDir(runTime.path()/masterInstDir/meshSubDir);
        Pstream::gatherList(haveMesh);
        Pstream::scatterList(haveMesh);

        meshes.set
        (
            regioni,
            loadOrCreateMesh
            (
                IOobject
                (
                    regionName,
                    masterInstDir,
                    runTime,
                    IOobject::MUST_READ,
                    IOobject::AUTO_WRITE,
                    true
                )
            ).ptr()
        );
        fvMesh& mesh = meshes[regioni];

        // loadOrCreateMesh's dummy zones can survive into the mesh, and
        // the unoriented dummyFaceZone fails a later zone sync. They never
        // coexist with real zones, so removing them by name is safe.
        if (mesh.faceZones().findIndex("dummyFaceZone") != -1)
        {
            Pout<< "DMR-GROW[r" << Pstream::myProcNo()
                << "] purging dummy zones after loadOrCreateMesh" << endl;
            mesh.pointZones().clear();
            mesh.faceZones().clear();
            mesh.cellZones().clear();
        }
    }

    // Phase 2: stitch the regions from geometry; the maps on disk are
    // sized for the old world and are not loaded
    const bool stitched = dmrHasNonConformalPatches(meshes);
    if (stitched)
    {
        dmrPostConstructAll(meshes);
        dmrConnectAll(regionNames, meshes);
    }

    // Phase 3: the fields, read stitched, and registered so the
    // redistribution transports them
    forAll(regionNames, regioni)
    {
        const boolList& haveMesh = haveMeshes[regioni];
        fvMesh& mesh = meshes[regioni];

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

        dmrReadFieldsGrow(haveMesh, mesh, allObjects, volScalars);
        dmrReadFieldsGrow(haveMesh, mesh, allObjects, volVectors);
        dmrReadFieldsGrow(haveMesh, mesh, allObjects, volSphereTensors);
        dmrReadFieldsGrow(haveMesh, mesh, allObjects, volSymmTensors);
        dmrReadFieldsGrow(haveMesh, mesh, allObjects, volTensors);
        dmrReadFieldsGrow(haveMesh, mesh, allObjects, surfScalars);
        dmrReadFieldsGrow(haveMesh, mesh, allObjects, surfVectors);
        dmrReadFieldsGrow(haveMesh, mesh, allObjects, surfSphereTensors);
        dmrReadFieldsGrow(haveMesh, mesh, allObjects, surfSymmTensors);
        dmrReadFieldsGrow(haveMesh, mesh, allObjects, surfTensors);

        pointMesh& pMesh = const_cast<pointMesh&>(pointMesh::New(mesh));

        PtrList<pointScalarField>          pointScalars;
        PtrList<pointVectorField>          pointVectors;
        PtrList<pointSphericalTensorField> pointSphereTensors;
        PtrList<pointSymmTensorField>      pointSymmTensors;
        PtrList<pointTensorField>          pointTensors;

        dmrReadFieldsGrow(haveMesh, pMesh, allObjects, pointScalars);
        dmrReadFieldsGrow(haveMesh, pMesh, allObjects, pointVectors);
        dmrReadFieldsGrow(haveMesh, pMesh, allObjects, pointSphereTensors);
        dmrReadFieldsGrow(haveMesh, pMesh, allObjects, pointSymmTensors);
        dmrReadFieldsGrow(haveMesh, pMesh, allObjects, pointTensors);

        dmrStoreFields(volScalars);
        dmrStoreFields(volVectors);
        dmrStoreFields(volSphereTensors);
        dmrStoreFields(volSymmTensors);
        dmrStoreFields(volTensors);
        dmrStoreFields(surfScalars);
        dmrStoreFields(surfVectors);
        dmrStoreFields(surfSphereTensors);
        dmrStoreFields(surfSymmTensors);
        dmrStoreFields(surfTensors);
        dmrStoreFields(pointScalars);
        dmrStoreFields(pointVectors);
        dmrStoreFields(pointSphereTensors);
        dmrStoreFields(pointSymmTensors);
        dmrStoreFields(pointTensors);
    }

    // Phase 4: conform, as fvMeshDistribute requires
    if (stitched)
    {
        dmrConformAll(regionNames, meshes);
    }

    // Phase 4b: zero-sized fields for the new ranks and the motion
    // reference, now that every region is conformal
    forAll(regionNames, regioni)
    {
        const word& regionName = regionNames[regioni];
        const boolList& haveMesh = haveMeshes[regioni];
        fvMesh& mesh = meshes[regioni];
        pointMesh& pMesh = const_cast<pointMesh&>(pointMesh::New(mesh));

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

            // Interpolating a point field constructs the subset's
            // pointMesh, which is collective, but only the master
            // interpolates. Construct it here on every rank first.
            (void)pointMesh::New(subsetterPtr().subMesh());

            dmrSendZeroFieldsGrow<volScalarField>(haveMesh, mesh, subsetterPtr());
            dmrSendZeroFieldsGrow<volVectorField>(haveMesh, mesh, subsetterPtr());
            dmrSendZeroFieldsGrow<volSphericalTensorField>(haveMesh, mesh, subsetterPtr());
            dmrSendZeroFieldsGrow<volSymmTensorField>(haveMesh, mesh, subsetterPtr());
            dmrSendZeroFieldsGrow<volTensorField>(haveMesh, mesh, subsetterPtr());
            dmrSendZeroFieldsGrow<surfaceScalarField>(haveMesh, mesh, subsetterPtr());
            dmrSendZeroFieldsGrow<surfaceVectorField>(haveMesh, mesh, subsetterPtr());
            dmrSendZeroFieldsGrow<surfaceSphericalTensorField>(haveMesh, mesh, subsetterPtr());
            dmrSendZeroFieldsGrow<surfaceSymmTensorField>(haveMesh, mesh, subsetterPtr());
            dmrSendZeroFieldsGrow<surfaceTensorField>(haveMesh, mesh, subsetterPtr());
            dmrSendZeroFieldsGrow<pointScalarField>(haveMesh, pMesh, subsetterPtr());
            dmrSendZeroFieldsGrow<pointVectorField>(haveMesh, pMesh, subsetterPtr());
            dmrSendZeroFieldsGrow<pointSphericalTensorField>(haveMesh, pMesh, subsetterPtr());
            dmrSendZeroFieldsGrow<pointSymmTensorField>(haveMesh, pMesh, subsetterPtr());
            dmrSendZeroFieldsGrow<pointTensorField>(haveMesh, pMesh, subsetterPtr());
        }

        PtrList<pointVectorField> pointVectors;

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

        dmrStoreFields(pointVectors);
    }

    // Phase 5: redistribute every region
    forAll(regionNames, regioni)
    {
        fvMesh& mesh = meshes[regioni];

        const bool addedPatch = dmrAddTransientInternalPatch(mesh);

        autoPtr<decompositionMethod> distributor
        (
            decompositionMethod::NewDistributor
            (
                decompositionMethod::decomposeParDict(runTime)
            )
        );
        const labelList finalDecomp = dmrDecompose(distributor(), mesh);

        // Same invariant as the shrink path (see dmrRedistributeToClone
        // for the full account): never hand fvMeshDistribute stale
        // geometry slices. Grow empties no rank, so it is not known to
        // fault, but the guard costs only a demand-driven recompute and
        // removes any reliance on that asymmetry.
        mesh.clearGeom();

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
    }

    // Phase 6: re-stitch on the new layout, fill the coupling faces and
    // write, map included (its offset tables are this world's)
    if (stitched)
    {
        dmrConnectAll(regionNames, meshes);

        forAll(regionNames, regioni)
        {
            dmrFillAfterStitch
            (
                meshes[regioni], regionNames[regioni], timeName, world.nNew
            );
        }
    }

    forAll(regionNames, regioni)
    {
        fvMesh& mesh = meshes[regioni];
        const fileName& meshSubDir = meshSubDirs[regioni];
        const fileName& masterInstDir = masterInstDirs[regioni];
        const boolList& haveMesh = haveMeshes[regioni];

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

        // On the new ranks loadOrCreateMesh wrote a dummy mesh with dummy
        // zones at the staged instance, which a later read could find.
        // Remove it, or only its zone files if it is this time's mesh.
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

    // The meshes leave the registry before the solver builds its own
    meshes.clear();

    // Old-layout caches go before the swap meshes, so a relaunch after a
    // failure there never reads them. The map just written is kept.
    dmrSweepStitchCaches(runTime, timeName, regionNames, stitched);

    dmrRedistributeSwapMeshes(runTime, Pstream::nProcs());

    // This is now the restart state
    if (dmrPurgeCheckpointsEnabled())
    {
        dmrPurgeSupersededCheckpoints(runTime, timeName);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        // Consume the meta so a later launch does not take it for a grow
        // in progress
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


// Values on the coupling faces after the re-stitch. The stitcher resizes
// patch fields without mapping values, so every non-conformal patch
// field holds uninitialised memory. The solver normally maps them back
// from conformed wrappers; the clones never had any. So each kind of
// field gets the value upstream itself uses when there is nothing to map
// from:
//
//   - volume fields: the adjacent cell value (as in
//     conformedFvPatchField::unconform). Mixed and fixed-gradient
//     conditions also get consistent refValue/refGrad/valueFraction or
//     gradient, which their constructors read at restart.
//   - face velocities (Uf, rhoUf): the interpolated volume velocity (as
//     in meshToMesh::interpolateUfs).
//   - fluxes (phi): Sf & Uf (as the solvers do after a topology change);
//     the first pressure corrector rewrites it.
//
// Any other surface field on coupling faces stops the run by name
// rather than being written uninitialised.
template<class Type>
static void dmrFillNonConformalVolFields(fvMesh& mesh)
{
    UPtrList<VolField<Type>> fields(mesh.fields<VolField<Type>>());

    forAll(fields, i)
    {
        typename VolField<Type>::Boundary& bf =
            fields[i].boundaryFieldRefNoStoreOldTimes();

        forAll(bf, patchi)
        {
            fvPatchField<Type>& pf = bf[patchi];

            if (!isA<nonConformalFvPatch>(pf.patch()))
            {
                continue;
            }

            pf == pf.patchInternalField();

            if (isA<mixedFvPatchField<Type>>(pf))
            {
                mixedFvPatchField<Type>& mpf =
                    refCast<mixedFvPatchField<Type>>(pf);
                mpf.refValue() = pf;
                mpf.refGrad() = Zero;
                mpf.valueFraction() = 1;
            }
            else if (isA<fixedGradientFvPatchField<Type>>(pf))
            {
                refCast<fixedGradientFvPatchField<Type>>(pf).gradient() =
                    Zero;
            }
        }
    }
}


static void dmrFillNonConformalFaceVelocities(fvMesh& mesh)
{
    UPtrList<surfaceVectorField> Ufs(mesh.fields<surfaceVectorField>());

    forAll(Ufs, i)
    {
        surfaceVectorField& Uf = Ufs[i];

        tmp<surfaceVectorField> tUfi;

        const volVectorField& U = surfaceToVolVelocity(Uf);

        if (notNull(U) && U.dimensions() == Uf.dimensions())
        {
            tUfi = fvc::interpolate(U);
        }
        else if
        (
            Uf.dimensions() == dimDensity*dimVelocity
         && mesh.foundObject<volScalarField>("rho")
         && mesh.foundObject<volVectorField>("U")
        )
        {
            tUfi = fvc::interpolate
            (
                mesh.lookupObject<volScalarField>("rho")
               *mesh.lookupObject<volVectorField>("U")
            );
        }
        else
        {
            FatalErrorInFunction
                << "cannot reconstruct the face velocity " << Uf.name()
                << " of region " << mesh.name()
                << " on the non-conformal faces: no volume velocity of"
                   " dimensions " << Uf.dimensions() << " found"
                << exit(FatalError);
        }

        surfaceVectorField::Boundary& bf =
            Uf.boundaryFieldRefNoStoreOldTimes();

        forAll(bf, patchi)
        {
            if (isA<nonConformalFvPatch>(bf[patchi].patch()))
            {
                bf[patchi] == tUfi().boundaryField()[patchi];
            }
        }
    }
}


static void dmrFillNonConformalFluxes(fvMesh& mesh)
{
    UPtrList<surfaceScalarField> fluxes(mesh.fields<surfaceScalarField>());
    UPtrList<surfaceVectorField> Ufs(mesh.fields<surfaceVectorField>());

    forAll(fluxes, i)
    {
        surfaceScalarField& phi = fluxes[i];

        const surfaceVectorField* UfPtr = nullptr;

        forAll(Ufs, j)
        {
            if (Ufs[j].dimensions()*dimArea == phi.dimensions())
            {
                UfPtr = &Ufs[j];
                break;
            }
        }

        if (!UfPtr)
        {
            FatalErrorInFunction
                << "cannot reconstruct the flux " << phi.name()
                << " of region " << mesh.name()
                << " on the non-conformal faces: no face velocity of"
                   " dimensions " << phi.dimensions()/dimArea << " found"
                << exit(FatalError);
        }

        surfaceScalarField::Boundary& bf =
            phi.boundaryFieldRefNoStoreOldTimes();

        forAll(bf, patchi)
        {
            if (isA<nonConformalFvPatch>(bf[patchi].patch()))
            {
                bf[patchi] ==
                (
                    mesh.Sf().boundaryField()[patchi]
                  & UfPtr->boundaryField()[patchi]
                );
            }
        }
    }
}


template<class Type>
static void dmrCheckNoNonConformalSurfaceFields(fvMesh& mesh)
{
    UPtrList<SurfaceField<Type>> fields(mesh.fields<SurfaceField<Type>>());

    if (fields.size())
    {
        FatalErrorInFunction
            << "surface field " << fields[0].name() << " of region "
            << mesh.name() << " has no rule for its non-conformal faces"
            << exit(FatalError);
    }
}


// Make the polyFaces map loadable by the new world. A mapped wall's entry
// carries a per-processor offset table sized for the world that stitched
// it, i.e. the old one, and the restart sends to every entry in it. The
// ranks beyond nNew hold no cells of the clone, so their blocks are
// empty and the table is cut at nNew. polyFacesBfRef() is private to
// fvMesh, hence the cast on a clone this code owns.
static void dmrTruncateMappedProcOffsets(fvMesh& mesh, const label nNew)
{
    const surfaceLabelField::Boundary& bf = mesh.polyFacesBf();

    forAll(bf, patchi)
    {
        if (!isA<nonConformalMappedPolyFacesFvsPatchLabelField>(bf[patchi]))
        {
            continue;
        }

        labelList& offsets =
            refCast<nonConformalMappedPolyFacesFvsPatchLabelField>
            (
                const_cast<fvsPatchLabelField&>(bf[patchi])
            ).procOffsets();

        if (offsets.size() <= nNew)
        {
            continue;
        }

        for (label proci = nNew; proci < offsets.size(); ++proci)
        {
            if (offsets[proci] != bf[patchi].size())
            {
                FatalErrorInFunction
                    << "polyFaces of patch " << bf[patchi].patch().name()
                    << " of region " << mesh.name() << " has "
                    << bf[patchi].size() - offsets[proci]
                    << " faces in the block of rank " << proci
                    << ", which the new world of " << nNew
                    << " ranks does not have" << exit(FatalError);
            }
        }

        offsets.setSize(nNew);
    }
}


static bool dmrHasNonConformalPatches(const PtrList<fvMesh>& clones)
{
    forAll(clones, i)
    {
        forAll(clones[i].boundary(), patchi)
        {
            if (isA<nonConformalFvPatch>(clones[i].boundary()[patchi]))
            {
                return true;
            }
        }
    }
    return false;
}


// A stationary stitcher on every mesh: no changers, no zones, and no
// connect yet (the changing stitcher needs libfvMeshStitchers and a
// dynamicMeshDict). Once per mesh.
static void dmrPostConstructAll(PtrList<fvMesh>& meshes)
{
    forAll(meshes, i)
    {
        meshes[i].postConstruct(false, false, fvMesh::stitchType::none);
    }
}


// Stitch the regions together from geometry, the startup form
// (changing=false, no map loaded). Collective; the meshes must be
// registered under their region names so the mapped walls find their
// neighbours. connect() on a region also connects its coupled
// neighbours, so later calls in the loop are no-ops on connected meshes.
static void dmrConnectAll(const wordList& regionNames, PtrList<fvMesh>& meshes)
{
    forAll(meshes, i)
    {
        fvMesh& mesh = meshes[i];

        if (!mesh.stitcher().stitches())
        {
            continue;
        }

        mesh.stitcher().connect(false, true, false);

        if (mesh.conformal())
        {
            FatalErrorInFunction
                << "region " << regionNames[i]
                << " is still conformal after the re-stitch"
                << exit(FatalError);
        }
    }
}


// The reverse: conform every region, as fvMeshDistribute requires. The
// coupling-face values are dropped and rebuilt after the re-stitch, as
// on the shrink side.
static void dmrConformAll(const wordList& regionNames, PtrList<fvMesh>& meshes)
{
    forAll(meshes, i)
    {
        fvMesh& mesh = meshes[i];

        if (!mesh.stitcher().stitches())
        {
            continue;
        }

        mesh.stitcher().disconnect(false, true);

        if (!mesh.conformal())
        {
            FatalErrorInFunction
                << "region " << regionNames[i]
                << " is still non-conformal after conforming"
                << exit(FatalError);
        }
    }
}


// After the re-stitch on the new layout: the coupling faces' values, the
// map's offset tables, and the map's write instance
static void dmrFillAfterStitch
(
    fvMesh& mesh,
    const word& regionName,
    const std::string& timeName,
    const label nNew
)
{
    if (!mesh.stitcher().stitches())
    {
        return;
    }

    // Volume fields first: the face velocities interpolate their
    // boundary values, and the fluxes use the face velocities
    dmrFillNonConformalVolFields<scalar>(mesh);
    dmrFillNonConformalVolFields<vector>(mesh);
    dmrFillNonConformalVolFields<sphericalTensor>(mesh);
    dmrFillNonConformalVolFields<symmTensor>(mesh);
    dmrFillNonConformalVolFields<tensor>(mesh);
    dmrFillNonConformalFaceVelocities(mesh);
    dmrFillNonConformalFluxes(mesh);
    dmrCheckNoNonConformalSurfaceFields<sphericalTensor>(mesh);
    dmrCheckNoNonConformalSurfaceFields<symmTensor>(mesh);
    dmrCheckNoNonConformalSurfaceFields<tensor>(mesh);

    dmrTruncateMappedProcOffsets(mesh, nNew);

    // Marks the map AUTO_WRITE at this instance, so mesh.write()
    // emits it and the restart's connect(load=true) finds it
    mesh.setPolyFacesBfInstance(timeName);

    Info<< "DMR: re-stitched region " << regionName
        << " on the new layout" << endl;
}


// Re-stitch the redistributed clones on the new layout before writing, so
// what is written is what the solver itself would write. Returns false,
// touching nothing, without non-conformal patches.
//
// The redistribution moves the conformal state, so a plain write leaves
// zero-size values on the non-conformal patches, while a restarted mesh
// stitches itself before reading fields and expects the coupled sizes.
// OpenFOAM's own load balancer re-stitches at the next move(); this does
// the same before the write.
//
// The startup form, connect(changing=false), only resizes the patch
// fields; the values are filled afterwards (dmrFillAfterStitch). The
// polyFaces map is written too, with its offset tables cut to the new
// world: a restart without a loadable map fails at its second mesh
// update.
//
// Mapped walls find their neighbour region by name in the registry, so
// the live meshes are checked out and the clones checked in for the
// duration (acceptable only because this group is about to exit).
static bool dmrRestitchClones
(
    const wordList& regionNames,
    UPtrList<fvMesh>& lives,
    PtrList<fvMesh>& clones,
    const std::string& timeName,
    const label nNew
)
{
    // Patch TYPES only, so consistent across ranks
    if (!dmrHasNonConformalPatches(clones))
    {
        return false;
    }

    dmrPostConstructAll(clones);

    // Region names now resolve to the redistributed meshes. Qualified
    // with regIOobject:: because objectRegistry's checkIn/checkOut
    // overloads hide the self-registration ones.
    forAll(clones, i)
    {
        lives[i].regIOobject::checkOut();
        if (!clones[i].regIOobject::checkIn())
        {
            FatalErrorInFunction
                << "could not register the redistributed mesh for region "
                << regionNames[i] << exit(FatalError);
        }
    }

    dmrConnectAll(regionNames, clones);

    forAll(clones, i)
    {
        dmrFillAfterStitch(clones[i], regionNames[i], timeName, nNew);
    }

    return true;
}


// The grow hook runs before the solver is selected, so the libraries
// that define the fields' boundary conditions (e.g. coupledTemperature)
// are not loaded yet, and those conditions would be read as generic
// patch fields that cannot be resized. Load what solver::New will load,
// from the same controlDict entries foamRun and foamMultiRun read.
static void dmrLoadSolverLibraries(const Time& runTime)
{
    const dictionary& controlDict = runTime.controlDict();

    wordHashSet solverNames;

    if (controlDict.isDict("regionSolvers"))
    {
        const dictionary& regionSolvers = controlDict.subDict("regionSolvers");

        forAllConstIter(dictionary, regionSolvers, iter)
        {
            if (iter().isStream())
            {
                solverNames.insert(word(iter().stream()));
            }
        }
    }
    else if (controlDict.found("solver"))
    {
        solverNames.insert(controlDict.lookup<word>("solver"));
    }

    forAllConstIter(wordHashSet, solverNames, iter)
    {
        solver::load(iter.key());
    }
}


static void dmrRestoreRegistry(UPtrList<fvMesh>& lives, PtrList<fvMesh>& clones)
{
    forAll(clones, i)
    {
        clones[i].regIOobject::checkOut();
        lives[i].regIOobject::checkIn();
    }
}


// Shrink, on the old group:
// 1. Checkpoint everything but the fields (writeNow).
// 2. Set decomposeParDict for the new world size.
// 3. Redistribute every live region into a clone in memory, re-stitch
//    the clones on the new layout if the case has non-conformal
//    couplings, and write them. Then the swap meshes.
// 4. Remove processor{N_new..N_old-1}, which the new group never reads.
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

    // Live region names, taken once before anything changes the registry
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

    // Checkpoint without the fields, which the redistribution writes in
    // the new layout (see dmrDemoteCheckpointFields)
    DynamicList<regIOobject*> demoted;
    forAll(regionNames, i)
    {
        dmrDemoteCheckpointFields
        (
            runTime.lookupObject<fvMesh>(regionNames[i]),
            demoted
        );
    }

    const double writeStart = MPI_Wtime();

    runTime.writeNow();

    const double writeEnd = MPI_Wtime();

    forAll(demoted, i)
    {
        demoted[i]->writeOpt() = IOobject::AUTO_WRITE;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        char writeElapsed[32];
        std::snprintf
        (
            writeElapsed, sizeof(writeElapsed), "%.3f", writeEnd - writeStart
        );

        dmrLifecycleLog
        (
            casePath,
            "Checkpoint at t=" + timeName
          + (isDmrForced ? " (DMR-forced)" : " (writeInterval)")
          + " in " + writeElapsed + " s ("
          + std::to_string(demoted.size()) + " field(s) deferred to the"
            " redistribute)"
        );
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
            "distributor", dmrDistributorName
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const double redistStart = MPI_Wtime();

    // Every region into a clone from memory; all clones are kept until
    // written, since the re-stitch needs all regions at once
    UPtrList<fvMesh> lives(regionNames.size());
    forAll(regionNames, i)
    {
        lives.set(i, &runTime.lookupObjectRef<fvMesh>(regionNames[i]));
    }

    PtrList<fvMesh> clones(regionNames.size());
    forAll(regionNames, i)
    {
        clones.set
        (
            i,
            dmrRedistributeToClone
            (
                runTime, timeName, regionNames[i], lives[i], 0
            ).ptr()
        );
    }

    const bool restitched =
        dmrRestitchClones(regionNames, lives, clones, timeName, world.nNew);

    forAll(clones, i)
    {
        clones[i].write();
        dmrWritePoints0Constant(clones[i]);
    }

    if (restitched)
    {
        dmrRestoreRegistry(lives, clones);
    }
    clones.clear();

    const double redistEnd = MPI_Wtime();

    // Old-layout caches go before the swap meshes, so a relaunch after a
    // failure there never reads them. The map just written is kept.
    dmrSweepStitchCaches(runTime, timeName, regionNames, restitched);

    // Before the retired ranks' directories go: the step is collective
    // and each rank clears its own stale pieces
    dmrRedistributeSwapMeshes(runTime, world.nNew);

    // The new group only reads processor0..nNew-1
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

    // This is now the restart state
    if (dmrPurgeCheckpointsEnabled())
    {
        dmrPurgeSupersededCheckpoints(runTime, timeName);
    }

    // Every rank is done; only now may the next group start from this
    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        dmrWriteWorldMeta(casePath, world);

        char elapsed[32];
        std::snprintf(elapsed, sizeof(elapsed), "%.3f", redistEnd - redistStart);

        dmrLifecycleLog
        (
            casePath,
            "Shrink redistribute complete ("
          + std::to_string(regionNames.size()) + " region(s)) in "
          + elapsed + " s."
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void dmrCheckpoint(Time& runTime, bool allRegions)
{
    // dmr_reconfigure follows and ends this process group, so running
    // asynchronous writers are reaped on the way out (a scope guard, for
    // every return path). Not on the way in: they write an earlier time
    // and can overlap this work.
    struct WriterJoin
    {
        ~WriterJoin() { dmrWaitForWriters(); }
    } writerJoin;

    // The world meta is the next group's proof that this reconfiguration
    // completed: it is written only once the data is in place, so any
    // meta from an earlier reconfiguration goes first. A group launched
    // after this one fails then finds none and stops (dmrRestart) instead
    // of running on a partial layout.
    if (Pstream::master())
    {
        Foam::rm(runTime.globalPath()/"log.dmr"/".world.meta");
    }

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
        // The new group has the max(N_old, N_new) ranks, so the data moves
        // there (dmrGrowRedistribute). This group only checkpoints.
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

    // Unknown direction: there is no correct target layout, so checkpoint
    // and stop. Known cause: dmr_get_analytics() reports only the latest
    // event, and the CE policy can overwrite the start-of-reconfiguration
    // event with DMR_EVENT_TALP_CHECK_CE_ACC.
    runTime.writeNow();

    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        const std::string casePath = runTime.globalPath();

        dmrLifecycleLog
        (
            casePath,
            "Checkpoint at t=" + runTime.name()
          + (isDmrForced ? " (DMR-forced)" : " (writeInterval)")
        );
        dmrLifecycleLog
        (
            casePath,
            "FATAL: reconfiguration direction unknown"
            " (DMR analytics unavailable)."
        );
    }

    FatalErrorInFunction
        << "DMR reconfiguration direction could not be determined." << nl
        << "The checkpoint at this time has been written; the run is"
        << " stopping rather than continuing with an unknown target"
        << " layout." << nl
        << "dmrQueryWorldInfo reads dmr_get_analytics(), which reports"
        << " only the latest event; the CE policy can overwrite the"
        << " start-of-reconfiguration event before this point."
        << exit(FatalError);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void dmrRestart(const std::string& casePath, bool allRegions)
{
    // Direction and sizes written by the old group. Collective.
    DmrWorldInfo world;
    const bool worldOk = dmrReadWorldMeta(casePath, world);

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
                "Restart: no .world.meta, so the previous group did not"
                " complete its reconfiguration."
            );
        }

        dmrUpdateDictEntry
        (
            casePath, "system/decomposeParDict",
            "numberOfSubdomains", std::to_string(newSize)
        );

        // Geometric decomposers need coefficients that no longer hold
        // once the process count changes
        dmrUpdateDictEntry
        (
            casePath, "system/decomposeParDict",
            "decomposer", "scotch"
        );

        // NewDistributor reads its own key
        dmrUpdateDictEntry
        (
            casePath, "system/decomposeParDict",
            "distributor", dmrDistributorName
        );

        dmrUpdateDictEntry
        (
            casePath, "system/controlDict",
            "startFrom", "latestTime"
        );

        // TODO: per-region system/<region>/decomposeParDict overrides are
        // not handled

        if (worldOk && world.dir == DmrDir::Shrink)
        {
            // The old group already wrote processor0..N_new-1
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
            // No meta: the old group failed before finishing, so the data
            // on disk may be in neither layout (see dmrCheckpoint)
            dmrLifecycleLog
            (
                casePath,
                "FATAL: the previous reconfiguration did not complete;"
                " not starting on a partial layout."
            );

            std::fprintf
            (
                stderr,
                "DMR: FATAL - the previous reconfiguration did not"
                " complete; not starting on a partial layout of '%s'."
                " See the previous group's error above.\n",
                casePath.c_str()
            );
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Grow: every rank must resolve the same latestTime, so each new rank
    // gets an empty restart-time directory; mesh and fields arrive in
    // dmrGrowRedistribute.
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

void dmrFinalize(const std::string& casePath, bool /*allRegions*/)
{
    // The last output time must not be left half written
    dmrWaitForWriters();

    MPI_Barrier(MPI_COMM_WORLD);

    // The output is already in the MIN layout, readable as a decomposed
    // case, so there is nothing to reconstruct (a serial reconstructPar
    // here would be the rank-0 gather this design avoids)
    if (Pstream::master())
    {
        dmrLifecycleLog
        (
            casePath,
            "Finalize: output already written in the MIN layout;"
            " no reconstruction needed."
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
