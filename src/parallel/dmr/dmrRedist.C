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
#include "CompactListList.H"

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
        const DMRStatus st = dmr_get_analytics(&a);

        if (st == DMR_SUCCESS)
        {
            // DMR changed DMRAnalytics::event from an enum (dmr_event_t)
            // to a string (const char*), which also shifts every field
            // after it. Compiling against one variant and linking the
            // other yields a silently corrupt struct: a truncated
            // pointer read as an enum, and node_count holding what is
            // really world_size (observed on MN5, 2026-07-26 — the
            // system dmr/2.0.1-dlb module vs the Slurm4DMR dmr-s4d
            // build). The string variant #defines the event names, so
            // that is a reliable compile-time discriminator between the
            // two APIs.
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

        // The direction MUST be resolved here: a shrink has to run on the
        // old (larger) group, because fvMeshDistribute needs max(N,N')
        // ranks, so it cannot be deferred to the new group the way a grow
        // can. When it comes back Unknown the run stops, so record enough
        // to tell WHY rather than guessing later: whether the analytics
        // call itself failed, or whether it succeeded with an event this
        // switch does not handle (the CE policy emits events the ROUND
        // policy never does, e.g. DMR_EVENT_TALP_CHECK_CE_ACC).
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


// Never let a partition boundary fall across a zero-area face.
//
// fvPatch::nf() is Sf()/mag(Sf()). It is evaluated on every COUPLED patch,
// so a face with mag(Sf) == 0 is harmless while it stays internal and
// divides by zero the moment it becomes a processor face. That is what
// fvMeshDistribute::distribute() does at the end of a redistribute, via
// correctCoupledPatchFields -> deltaCoeffs -> fvPatch::delta() -> nf().
//
// Axisymmetric (wedge) meshes have exactly such faces: the cell face on
// the axis collapses to a line as r -> 0. The solver never trips over
// them because it only needs deltaCoeffs on faces that are not processor
// faces; only a re-cut promotes an axis face to a processor face, which
// is why this appears at a reconfiguration and not during time stepping,
// and why it is intermittent (it needs the partitioner to cut there).
//
// Rather than repair the geometry or guard the division, both of which
// are OpenFOAM's own code, forbid the cut: any two cells sharing a
// degenerate face are pinned to the same destination. Union-find, so a
// chain of collapsed faces collapses to one group. A no-op on meshes that
// have no such faces, which is a single pass over the internal faces.
static label dmrPinDegenerateFaces(const fvMesh& mesh, labelList& decomp)
{
    const label nCells = mesh.nCells();
    if (nCells == 0 || decomp.size() != nCells)
    {
        return 0;
    }

    const scalarField& magSf = mesh.magSf().primitiveField();
    const labelUList& own = mesh.faceOwner();
    const labelUList& nei = mesh.faceNeighbour();
    const label nInt = mesh.nInternalFaces();

    // Absolute, not relative to the mesh scale: the failure is division by
    // EXACTLY zero, so the threshold only has to separate a true zero (and
    // anything that underflowed to one) from the smallest legitimate face.
    // A relative threshold would start pinning healthy small faces on
    // finely resolved meshes and quietly damage the load balance.
    const scalar tol = vSmall;

    labelList root(identityMap(nCells));

    // find with path halving
    auto find = [&root](label i)
    {
        while (root[i] != i)
        {
            root[i] = root[root[i]];
            i = root[i];
        }
        return i;
    };

    label nPinned = 0;
    for (label facei = 0; facei < nInt; ++facei)
    {
        if (magSf[facei] > tol)
        {
            continue;
        }

        const label a = find(own[facei]);
        const label b = find(nei[facei]);
        if (a != b)
        {
            root[a > b ? a : b] = (a < b ? a : b);
        }
        ++nPinned;
    }

    // Face-area census of the mesh we are about to redistribute, over
    // internal AND boundary faces. This is what says whether a zero-area
    // face is being HANDED to fvMeshDistribute or CREATED inside it, which
    // are different problems with different owners.
    {
        const scalarField& allMag = mesh.magSf();
        scalar mn = great, mx = 0;
        label nTiny = 0;
        forAll(allMag, i)
        {
            mn = min(mn, allMag[i]);
            mx = max(mx, allMag[i]);
            if (allMag[i] <= vSmall) ++nTiny;
        }
        scalar bMn = great;
        label nbTiny = 0, nbFaces = 0;
        forAll(mesh.boundary(), patchi)
        {
            const scalarField& pm = mesh.boundary()[patchi].magSf();
            forAll(pm, i)
            {
                bMn = min(bMn, pm[i]);
                if (pm[i] <= vSmall) ++nbTiny;
                ++nbFaces;
            }
        }
        Pout<< "DMR: faceArea census before distribute:"
            << " internal min=" << mn << " max=" << mx
            << " zero=" << nTiny
            << " | boundary faces=" << nbFaces
            << " min=" << (nbFaces ? bMn : 0) << " zero=" << nbTiny
            << " | pinned=" << nPinned << endl;
    }

    if (nPinned == 0)
    {
        return 0;
    }

    // One destination per group: the group representative's, which is its
    // lowest-numbered cell, so the choice does not depend on face order.
    forAll(decomp, celli)
    {
        const label r = find(celli);
        if (r != celli)
        {
            decomp[celli] = decomp[r];
        }
    }

    Pout<< "DMR: pinned " << nPinned
        << " zero-area face(s) (mag(Sf) <= " << tol
        << ") so no partition boundary crosses them" << endl;

    return nPinned;
}


// Run the distributor and return one destination per cell.
//
// ptscotch returns a SIZE-1 list for a ZERO-CELL mesh, which
// fvMeshDistribute rejects with
//     "Size of distribution:1 mesh nCells:0"   (fvMeshDistribute.C:1819)
// Zero-cell meshes are normal on the grow path: loadOrCreateMesh fabricates
// them on the ranks that hold no data yet, so every grow onto new ranks hit
// this (MN5, 2026-07-26, 448 -> 560). Serial scotch happens to return an
// empty list and so never showed the problem.
//
// decompose() is collective, so it cannot be skipped on the empty ranks -
// they must call it with everyone else. The result is corrected afterwards
// instead: a mesh with no cells has, trivially, an empty distribution.
static labelList dmrDecompose
(
    decompositionMethod& distributor,
    const fvMesh& mesh
)
{
    labelList decomp(distributor.decompose(mesh, mesh.cellCentres()));

    if (mesh.nCells() == 0 && decomp.size() != 0)
    {
        // Logged so a run can PROVE the guard is present and firing: its
        // absence in the output means an older libfoamDmr is loaded.
        Pout<< "DMR: distributor returned " << decomp.size()
            << " entries for a 0-cell mesh; truncating to 0" << endl;

        decomp.setSize(0);
    }

    dmrPinDegenerateFaces(mesh, decomp);

    return decomp;
}


// Partitioning method used for the in-process redistributes. ptscotch is
// the parallel one: it registers ONLY as a distributor, which is exactly
// this job, whereas scotch also registers as decomposePar's serial
// decomposer. On MN5 a 2800 -> 1792 reconfiguration of a 100M-cell case
// sat in scotch's decompose() for over an hour without emitting anything
// (2026-07-26).
static const char* const dmrDistributorName = "ptscotch";


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
        // Transient: restored immediately after the write below. The
        // registry hands out const access, but the write option is
        // bookkeeping, not field data.
        GeoField& f = const_cast<GeoField&>(*fields[names[i]]);

        if (f.writeOpt() == IOobject::AUTO_WRITE)
        {
            f.writeOpt() = IOobject::NO_WRITE;
            demoted.append(&f);
        }
    }
}


// The reconfiguration checkpoint and the redistribute write the same
// fields twice: once in the old layout, then again in the new one. When
// the redistribute sources them from memory nothing ever reads that
// first copy, so it is pure duplicated I/O.
//
// It cannot simply be skipped: runTime.writeNow() also emits the state
// the redistribute does NOT produce, notably uniform/time (time index
// and deltaT, i.e. restart continuity). So instead of dropping the
// write, drop the fields FROM it — demote them here, write, restore.
// Everything else Time writes still goes out.
//
// Accepted consequence: with no old-layout field copy on disk, a
// redistribute that aborts leaves the previous natural write as the
// newest checkpoint, so the relaunch loses that one interval instead of
// resuming at the reconfiguration instant.
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


// Construct a fresh fvMesh from disk (non-registered, so it doesn't clash
// with the live mesh still in runTime's registry), load every field type
// into it, run fvMeshDistribute, and write the new layout. Mirrors the
// body of redistributePar.C's main() — calling fvMeshDistribute against
// this clean mesh avoids the receiver-side deadlock that happens when it
// is called against a mesh carrying live solver state. The region argument
// is the polyMesh name (polyMesh::defaultRegion for single-region cases,
// or one of the case's named regions for foamMultiRun).
//
// targetRanks narrows the DESTINATION width below the running world size.
// Zero, the reconfiguration paths, means "take numberOfSubdomains from the
// dict as it stands", i.e. redistribute across the whole world. A positive
// value, the uniform-write path, sends everything to the first targetRanks
// ranks and leaves the rest empty. Either way the operation is collective
// over the whole world and needs no ranks beyond the ones already running,
// because fvMeshDistribute's requirement is max(source, target) ranks and
// the source width is the world width in both cases.
static void dmrRunFreshRedistribute
(
    Time& runTime,
    const std::string& timeName,
    const word& regionName,
    const fvMesh& live,
    const label targetRanks = 0
)
{
    // Always from the live mesh in memory, never re-read from the
    // checkpoint on disk. The two callers both hold it: the shrink runs
    // in the DYING group, which is still resident, and the uniform write
    // runs in the solver itself. Grow is the one case with no live mesh
    // to clone (its group is a fresh set of processes and the ranks that
    // held the data are gone), and it has its own reader built for that,
    // loadOrCreateMesh, which tolerates ranks holding nothing.
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

    // Point fields (e.g. pointDisplacement on moving meshes). Their mesh
    // type is pointMesh, not fvMesh, so they clone against pointMesh::New.
    // fvMeshDistribute maps every field registered on the mesh, so these
    // must be cloned (and thereby registered) before distribute() runs.
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

    const bool addedPatch = dmrAddTransientInternalPatch(mesh);

    fvMeshDistribute(mesh).distribute(finalDecomp);

    if (addedPatch)
    {
        dmrRemoveTransientInternalPatch(mesh);
    }

    mesh.setInstance(timeName);
    mesh.write();

    // constant/<region>/polyMesh/points is the motion reference: SHARED
    // with the run, not private to this output time. Rewriting it in the
    // written layout is right when the solver is about to be replaced
    // (the reconfiguration paths, targetRanks == 0), and is corruption at
    // a natural write, where those same processor directories keep
    // serving a solver that is still running at its own width. The
    // transported reference reaches the time directory either way, as the
    // points0 field mesh.write() has just emitted.
    if (targetRanks == 0)
    {
        dmrWritePoints0Constant(mesh);
    }
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


// * * * * * * * * * Uniform (MIN-layout) output write  * * * * * * * * * * //
//
// A malleable run writes its natural output in whatever layout it
// happens to be running at, so the time series ends up decomposed
// N ways at one time and N' ways at the next: not a readable case,
// because a decomposed case has ONE decomposition across all times.
//
// The fix is to emit every natural write in the layout the run is
// always able to produce, MIN = policy-min nodes x procs-per-node,
// without ever gathering onto rank 0. To keep the solver running while
// that happens, the write is handed to a SECOND MPI WORLD spawned for
// the purpose: the live world copies its state to it rank-for-rank,
// the helper redistributes to MIN and writes, then frees its memory and
// exits. The solver does not write this step itself and does not wait.
//
// The two worlds are the same size N because fvMeshDistribute needs
// max(N_src, N_dst) ranks and MIN <= N, so the handoff is a straight
// 1:1 rank-to-rank copy with no re-decomposition in flight. Nothing
// goes through disk on the way. Verified separately (2026-07-25) that
// a spawned process runs as a full OpenFOAM application with its own
// MPI_COMM_WORLD, so Pstream and fvMeshDistribute behave normally
// inside the helper; the parent intercommunicator is used only for the
// handoff, via raw MPI byte transfer.

// Target world size for the uniform output: policy-min NODES times
// procs-per-node. Taken from the environment because that is how the
// launcher configures DMR itself (DMR_DEFAULT_POLICY_MIN /
// DMR_PROCS_PER_NODE); the library exposes a setter for the policy
// minimum but no getter.
static int dmrMinWorldSize()
{
    const char* minNodes = std::getenv("DMR_DEFAULT_POLICY_MIN");
    const char* ppn = std::getenv("DMR_PROCS_PER_NODE");

    const int n = minNodes ? std::atoi(minNodes) : 1;
    const int p = ppn ? std::atoi(ppn) : 1;

    return (n > 0 ? n : 1) * (p > 0 ? p : 1);
}


// Emit every natural write in the MIN layout via a spawned helper, so
// the output series shares one decomposition and is directly readable.
// On by default; set FOAM_DMR_UNIFORM_WRITE=0 to write in the running
// layout instead (which leaves the series unreadable across
// reconfigurations, but needs no MPI_Comm_spawn).
static bool dmrUniformWriteEnabled()
{
    static const bool enabled = []()
    {
        const char* env = std::getenv("FOAM_DMR_UNIFORM_WRITE");
        return !env || std::string(env) != "0";
    }();
    return enabled;
}


// The intercommunicator of the helper world that is still writing, or
// MPI_COMM_NULL when none is outstanding. At most ONE helper is alive at
// a time (see dmrWaitForHelper).
static MPI_Comm dmrHelperComm = MPI_COMM_NULL;


// Wait for the outstanding helper to finish, if there is one.
//
// MPI_Comm_disconnect is collective over BOTH worlds and returns only
// once each side has called it. The helper calls it as its LAST action,
// so the parent calling it here blocks exactly until the helper has
// finished writing. That is the whole synchronisation mechanism: no
// extra protocol, no polling.
//
// The parent deliberately does NOT call this at the natural write - that
// would make the handoff synchronous again. It is called only where a
// still-running helper would otherwise be destroyed:
//
//   * before spawning a replacement (bounds the system to one helper,
//     so helpers cannot accumulate if writes outpace them);
//   * at every exit of dmrCheckpoint, because dmr_reconfigure() tears
//     down and restarts the DVM the helper lives in;
//   * at dmrFinalize, for the same reason at end of run.
//
// By then the parent has already done its own checkpoint and
// redistribute work, which is independent of the helper's, so in
// practice the helper finished long ago and this returns immediately.
static void dmrWaitForHelper()
{
    if (dmrHelperComm == MPI_COMM_NULL)
    {
        return;
    }

    // A helper that died leaves this collective unsatisfied. Ask for the
    // error to be returned rather than aborting the solver, so the
    // failure is diagnosable; it cannot rescue a genuinely hung peer,
    // which is why the helper also drops a .complete sentinel for
    // after-the-fact checking.
    //
    // No attempt is made to save and restore the previous handler: this
    // communicator is destroyed by the disconnect below, and the default
    // is a predefined handler that must not be freed.
    MPI_Comm_set_errhandler(dmrHelperComm, MPI_ERRORS_RETURN);

    const int rc = MPI_Comm_disconnect(&dmrHelperComm);

    if (rc != MPI_SUCCESS)
    {
        WarningInFunction
            << "MPI_Comm_disconnect from the uniform-write helper failed"
            << " (code " << rc << "). The output time it was writing may"
            << " be incomplete; check for its .complete sentinel." << endl;

        // On failure the handle may be left valid. Drop it rather than
        // retry: a second attempt would block on the same dead peer, and
        // hanging the solver is worse than leaking one communicator.
        dmrHelperComm = MPI_COMM_NULL;
    }
}


// Scope guard so the wait cannot be missed on an exit path added later:
// dmrCheckpoint returns from several places, and every one of them is
// immediately followed by dmr_reconfigure() killing the helper.
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


// Serialise this rank's slice of one field type into the fields blob,
// as a sub-dictionary per field. Only the AUTO_WRITE set is sent: it is
// exactly what a natural write would have emitted, and it keeps the
// live registry's derived/cached state (which differs between ranks)
// out of the helper, for the same reason the clone path filters it.
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

    // At most one helper at a time. If the previous one is somehow still
    // writing when the next output time arrives, wait for it here rather
    // than accumulating helper worlds (and their processes, and their
    // intercommunicators) without bound. Normally a no-op: the helper
    // finished many timesteps ago.
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

    // Provision the helper exactly the way DMR provisions its own spawned
    // processes (dmr_core.c, the MPI_Comm_spawn_multiple call): the
    // PMIX_* keys add the hosts and slots for the NEW processes and allow
    // them to overload the cores. They apply only to the spawned world, so
    // the solver's own placement is untouched and DMR's node accounting
    // (identify_nodes -> MPI_Comm_split_type on its INTERNAL_COMM_WORLD)
    // still sees the run exactly as before. Passing MPI_INFO_NULL instead
    // is what made the spawn fail with "all nodes already filled" once the
    // allocation was exactly filled by the solver.
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

    // A refused spawn must degrade to "this output time stays in the
    // running layout", never kill the solver: the default
    // MPI_ERRORS_ARE_FATAL would abort before the return code is read.
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
        // Quoted: a time name like "0.05" would otherwise read back as
        // a scalar rather than a word
        // 17 significant digits = max_digits10 for an IEEE double, the
        // minimum that round-trips a double through text exactly.
        // deltaT below sets the step the run resumes with, and
        // OpenFOAM's default writePrecision of 6 would round it.
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

    // Hand off and carry on. The intercommunicator is deliberately NOT
    // disconnected here: disconnect is collective and would block until
    // the helper finished, making the write synchronous again. It is
    // kept so the wait can happen later, at the one point where the
    // helper would otherwise be killed (see dmrWaitForHelper).
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


// A DMR-forced checkpoint is restart state, not output: it carries
// whatever layout the reconfiguration produced, so it is unreadable as
// part of the uniform output series, and at real mesh sizes the
// accumulated copies are a serious waste of disk.
//
// Mark each one with a sentinel and drop every SUPERSEDED one from this
// rank's own processor directory (no cross-rank collisions). The newest
// is always kept: it IS the state the next generation restarts from.
// Same policy as the serial path's sweep in dmrCheckpoint, applied to
// the processor trees. OpenFOAM time names must be ordered as floats,
// not as strings, or "0.17" sorts before "0.085".
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
    // The region set comes from the registry below rather than from this
    // flag: every live fvMesh has to be redistributed, because the
    // MIN-layout series is only readable as a case if all of them are.
    (void)allRegions;

    const label minRanks = dmrMinWorldSize();
    const std::string timeName = runTime.name();

    // Live meshes, in an order that is identical on every rank: everything
    // below is collective, so the regions must be visited in lockstep.
    wordList regionNames;
    {
        HashTable<fvMesh*> liveRegions(runTime.lookupClass<fvMesh>());
        regionNames = liveRegions.sortedToc();
    }

    // Preconditions, checked BEFORE the first collective so that a case
    // this path cannot serve degrades to an ordinary write rather than
    // half-writing. Past this point there is no fallback: a failure
    // inside fvMeshDistribute leaves the ranks at different points in a
    // collective sequence, and continuing from there is not recoverable.
    if (minRanks < 1 || regionNames.empty())
    {
        runTime.write();
        return;
    }

    // Redistribute a CLONE of each live mesh down to the MIN layout.
    //
    // This is the difference between a natural write and the forced
    // checkpoint DMR takes before a reconfiguration. There the solver is
    // about to be torn down, so the live mesh may be consumed. Here the
    // time loop continues immediately afterwards at the running width, so
    // the live mesh must come through untouched. dmrCloneLiveMesh gives a
    // separate, non-registered fvMesh built from the live mesh's
    // primitives; fvMeshDistribute only ever sees that clone, and the
    // clone dies with the autoPtr on return. The live mesh is neither
    // redistributed nor re-instanced nor written.
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

    // The non-field state, uniform/time, which the clones do not carry.
    //
    // NOT runTime.write(), and not runTime.writeNow(): in the helper
    // world those were safe because its registry held nothing but the
    // clones, whereas here runTime owns the LIVE meshes and their fields.
    // Either call would walk that registry and emit the running N-way
    // layout on top of the MIN-way layout just written, into the same
    // directories. writeTimeDict is the narrow part that is actually
    // wanted.
    runTime.writeTimeDict();

    // Derivable caches, as after any redistribute
    dmrSweepStitchCaches(runTime, timeName, regionNames);

    // Each rank only ever touches its OWN processor directory below, so
    // the three barriers are not there to order file accesses against one
    // another. Each buys something different:
    //
    //   1. the COLLATED file handler routes writes through master ranks
    //      and is genuinely collective, so a rank running ahead into the
    //      rmDir below could overtake a write still in flight. No-op
    //      under the default uncollated handler.
    MPI_Barrier(MPI_COMM_WORLD);

    // Ranks outside the MIN layout received no cells, so what they just
    // wrote is an empty mesh. Drop it, leaving the output series exactly
    // minRanks wide and readable as a case.
    if (Pstream::myProcNo() >= minRanks)
    {
        const fileName stale(runTime.path()/timeName);
        if (isDir(stale))
        {
            rmDir(stale);
        }
    }

    //   2. promotes what the sentinel MEANS, from "this rank's slice
    //      finished" to "the whole output time finished", which is the
    //      reading that makes it useful for rejecting a partial write.
    MPI_Barrier(MPI_COMM_WORLD);

    // Sentinel, written LAST by every rank that keeps this time: an
    // output directory without it was produced by a write that died
    // partway, so the time is suspect and should be discarded rather
    // than trusted.
    if (Pstream::myProcNo() < minRanks)
    {
        const fileName marker(runTime.path()/timeName/".complete");
        if (FILE* f = std::fopen(marker.c_str(), "w"))
        {
            std::fclose(f);
        }
    }

    //   3. keeps the write's skew OUT OF THE CE READING. Correctness does
    //      not need it: the next timestep's first linear solve is a
    //      collective that resynchronises anyway. But that is the whole
    //      problem. The tail above is uncollective filesystem work, so
    //      ranks leave at different times, and without a barrier that
    //      spread is absorbed by the next timestep's first collective,
    //      where TALP charges it to the solver instead of to the write.
    //      CE is what the controller reconfigures on, so a write must not
    //      be able to depress it.
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

    // On the uniform path the solver does its own I/O, but writes the
    // MIN-layout copy rather than its running layout: a plain
    // runTime.write() here would emit the N-way state into the very
    // directories the MIN-way state is going to, and the two would
    // interleave.
    if (uniform)
    {
        dmrUniformWriteInPlace(runTime, allRegions);
        return;
    }

    // At MIN width the running layout already IS the target, so this is
    // an ordinary write. It must still emit its own polyMesh.
    //
    // OpenFOAM resolves a time directory that carries no mesh by searching
    // backwards for the nearest one that does, and under the uniform-write
    // regime the nearest one is a mesh this run is not using: the uniform
    // write re-decomposes from scratch at every output time, whereas the
    // layout the solver is actually running at came from the last
    // reconfiguration. Both are MIN-wide, so nothing looks wrong, but they
    // assign different cells to each rank and give the patches different
    // sizes. The fields then read against the wrong partition, which
    // surfaces far away as a boundary field whose value list does not
    // match its patch. Writing the mesh makes every output time
    // self-contained, which is the invariant the whole series relies on.
    //
    // setInstance, not a bare write flag: it re-arms points AND topology
    // for writing (setPointsWrite/setTopologyWrite, which polyMesh clears
    // after each write so a static mesh is emitted once) and points them
    // at this time directory rather than the one the mesh was read from.
    // It touches only IOobject instances and write options, never
    // topology or geometry, so the running solver is unaffected.
    if (dmrUniformWriteEnabled() && runTime.writeTime())
    {
        HashTable<fvMesh*> liveRegions(runTime.lookupClass<fvMesh>());
        forAllIter(HashTable<fvMesh*>, liveRegions, iter)
        {
            iter()->setInstance(runTime.name());
        }
    }

    runTime.write();
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

    // Drain every blob up front so the transfer itself is over quickly.
    // The disconnect is deliberately left to the END of this function:
    // it is collective with the parent, and the parent uses it as its
    // "helper has finished" signal (dmrWaitForHelper). Concurrency comes
    // from WHEN THE PARENT WAITS - only at a reconfiguration, after its
    // own checkpoint and redistribute work - not from disconnecting
    // early here. Disconnecting early would leave this process writing
    // with no way for the parent to know, and the DVM teardown at
    // dmr_reconfigure() would kill it mid-write.
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
        const labelList finalDecomp = dmrDecompose(distributor(), mesh);

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

    // As on the shrink side: this is now the restart state, so older
    // forced checkpoints are superseded
    if (dmrUniformWriteEnabled())
    {
        dmrPurgeSupersededCheckpoints(runTime, timeName);
    }

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

    // Snapshot the live region names BEFORE anything writes or
    // redistributes. Each dmrRunFreshRedistribute builds a
    // non-registered fvMesh that never enters the registry, so taking
    // the live mesh table once up front avoids any
    // iterator-vs-mutation hazard.
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

    // The redistribute sources the fields from memory and rewrites every
    // one of them in the new layout, so writing them here in the old
    // layout first is redundant I/O; keep the rest of the checkpoint
    // (uniform/time above all).
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
            "distributor", dmrDistributorName
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const double redistStart = MPI_Wtime();

    forAll(regionNames, i)
    {
        // The live mesh is still resident on this (dying) group, so the
        // checkpoint it was just written from is reproduced in memory
        // rather than read back off disk
        dmrRunFreshRedistribute
        (
            runTime,
            timeName,
            regionNames[i],
            runTime.lookupObject<fvMesh>(regionNames[i])
        );
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

    // This checkpoint is now the restart state; older ones are dead
    // weight (only meaningful with the uniform write, which is what
    // makes the surviving natural writes the readable series)
    if (dmrUniformWriteEnabled())
    {
        dmrPurgeSupersededCheckpoints(runTime, timeName);
    }

    if (Pstream::master())
    {
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
    // Everything below is followed by dmr_reconfigure(), which tears down
    // and restarts the DVM the uniform-write helper lives in, killing it
    // mid-write. Wait for it before that happens - but on the way OUT, so
    // this rank's own checkpoint and redistribute (hundreds of seconds on
    // a large case) overlap the helper's write rather than serialising
    // with it. There is no data dependency between them: the helper is
    // writing a previous output time, this rank is handling the current
    // reconfiguration. Declared as a scope guard so it also covers the
    // early returns below, and any added later.
    const dmrHelperBarrier helperBarrier;

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

    // Unknown direction. Nothing correct is left to do: the redistribute
    // has to know which way the world is changing, and a checkpoint
    // written for the wrong target does not match the group that will
    // read it. Flush what we have, say why, and stop.
    //
    // dmr_get_analytics() reports only the LATEST event, and the CE
    // policy emits DMR_EVENT_TALP_CHECK_CE_ACC after the
    // start-of-reconfiguration event dmrQueryWorldInfo switches on. That
    // is the known way to arrive here.
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
            "distributor", dmrDistributorName
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
            // Both directions place the data themselves. There is no
            // third way to obtain it: only the direction says which group
            // holds it and which layout it is in.
            dmrLifecycleLog
            (
                casePath,
                "FATAL: reconfiguration direction unknown at restart"
                " (no world meta from the old group)."
            );

            std::fprintf
            (
                stderr,
                "DMR: FATAL - reconfiguration direction unknown at"
                " restart; refusing to guess the layout of '%s'.\n",
                casePath.c_str()
            );
            MPI_Abort(MPI_COMM_WORLD, 1);
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

void dmrFinalize(const std::string& casePath, bool /*allRegions*/)
{
    // End of run: the helper must finish before this process group exits,
    // or the last output time is left half written.
    dmrWaitForHelper();

    MPI_Barrier(MPI_COMM_WORLD);

    // Nothing to do. The run's output is already written in the uniform
    // MIN layout by the natural-write path, so it is directly readable as
    // a decomposed case: there is nothing to reconstruct. The serial
    // reconstructPar that used to run here was removed deliberately — it
    // is a rank-0 gather, exactly the bottleneck this pipeline exists to
    // eliminate, and its -rm plus the processor-directory cleanup that
    // followed it would delete the very series that was produced.
    // Superseded DMR-forced checkpoints are already dropped as they are
    // superseded (dmrPurgeSupersededCheckpoints), so nothing accumulates.
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
