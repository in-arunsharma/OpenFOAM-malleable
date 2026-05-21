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

void dmrCheckpoint(Time& runTime, bool allRegions)
{
    // DMR-forced checkpoints (off-cadence) get a sentinel file so the
    // sweep below can discard them once a newer checkpoint exists.
    const bool isDmrForced = !runTime.writeTime();

    // writeNow()'s bool return is the AND over every registered
    // regIOobject including transient dynamic-mesh sub-objects, so it
    // is not a useful success signal. Real I/O failures propagate via
    // FatalError and abort before we reach reconstructPar below.
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

        // For moving-mesh cases the polyMesh topology is only written at
        // mesh-swap times; between those, time directories contain only
        // a `points` file. decomposePar evolves topology forward via a
        // comma-separated time chain (e.g. -time 0.1,0.8) — see bug
        // 0004368 on bugs.openfoam.org. dmrBuildDecomposeTimeArg
        // collects the mesh-write times from the case and builds that
        // argument; no polyMesh files are copied.
        const std::string timeArg =
            dmrBuildDecomposeTimeArg(casePath, restartTime, allRegions);

        std::string cmd =
            "decomposePar -force -time " + timeArg + " -cellProc"
          + " -case '" + casePath + "'";
        if (allRegions) cmd += " -allRegions";

        dmrRunOrAbort(casePath, "dmr_decompose.log", cmd);

        // Engine-style cases keep additional sub-meshes under
        // constant/meshes/<name> that -allRegions does not visit.
        // Mirror the case's Allrun pattern: decompose each sub-mesh
        // explicitly for the new process count. No-op for cases
        // without sub-meshes.
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
                // No -force here: OpenFOAM rejects `-force` combined with
                // `-mesh` plus a single `-region` ("Cannot force the
                // decomposition of a single region").  The main
                // `decomposePar -force -allRegions` above has already
                // cleared the processor* dirs, so each sub-mesh decompose
                // can append its output without forcing.
                const std::string mcmd =
                    "decomposePar -mesh " + meshName
                  + " -region fluid -case '" + casePath + "'";
                dmrRunOrAbort(casePath, "dmr_decompose.log", mcmd);
            }
            globfree(&mg);
        }

        dmrLifecycleLog(casePath, "Decomposition complete.");
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

        std::string cmd =
            "reconstructPar -newTimes -rm -case '" + casePath + "'";
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
