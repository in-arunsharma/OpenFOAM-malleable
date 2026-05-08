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

// Whether DMR-internal subprocess logging is enabled.  The FOAM_ prefix
// marks this as an OpenFOAM-side hook control (vs the DMR_ namespace which
// belongs to the BSC DMR library itself).  Default on; set FOAM_DMR_LOG=0
// to silence the per-utility logs entirely.
static bool dmrLogEnabled()
{
    static const bool enabled = []()
    {
        const char* env = std::getenv("FOAM_DMR_LOG");
        return !env || std::string(env) != "0";
    }();
    return enabled;
}


// Resolve the redirect target for a single utility's output.  When logging
// is enabled, returns <case>/log.dmr/<logName>; otherwise /dev/null.  The
// folder name "log.dmr" is intentional: OpenFOAM's foamLog tool reserves
// "logs/" for extracted residual data, and the leading "log." matches the
// OF case-root convention (log.foamRun, log.blockMesh).
static std::string dmrLogPath
(
    const std::string& casePath,
    const std::string& logName
)
{
    if (!dmrLogEnabled()) return "/dev/null";
    return casePath + "/log.dmr/" + logName;
}


// Append one lifecycle event to log.dmr/dmr_lifecycle.log AND echo it
// to stderr.  This consolidates the DMR-OF lifecycle messages (checkpoint
// start/end, restart with N procs, finalize, ...) into a single file so
// the user has all DMR-side state alongside reconstruct/decompose logs,
// rather than scattered across slurm-N.out and slurm-N.err.  The stderr
// echo preserves live visibility.  Each line is wall-clock timestamped
// in UTC ISO-8601 form so it can be cross-referenced with the
// [DMR ANALYTICS] events emitted by the DMR library itself.
static void dmrLifecycleLog
(
    const std::string& casePath,
    const std::string& msg
)
{
    // Live echo to stderr, regardless of FOAM_DMR_LOG.  The whole point
    // of the lifecycle log is to surface what DMR is doing — silencing
    // it entirely would defeat the purpose.
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


// Run a single utility as a subprocess, append its output to <case>/log.dmr/.
// On non-zero exit:
//   abortOnFailure = true  → MPI_Abort(MPI_COMM_WORLD, 1)
//                            (used during checkpoint/restart, where a stale
//                             case would corrupt the next reconfig)
//   abortOnFailure = false → return the exit code, log a warning
//                            (used in dmrFinalize, where the simulation
//                             has already finished and the case state is
//                             still valid for inspection)
// Only called by master rank.
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


// Update a single entry in a case dictionary by calling foamDictionary.
// We tried doing this in-process via Foam::dictionary read/write, but the
// in-process round-trip silently produced a no-op write on multi-region
// controlDicts that contain a regionSolvers sub-dictionary plus a
// leading C++ comment block.  foamDictionary handles every dict format
// the user might bring uniformly, so it is safer to keep the subprocess
// even though it costs three fork+exec per reconfiguration.
//
// Aborts on subprocess failure: a stale dict at the start of the new
// process group would corrupt every subsequent reconfiguration.
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


// Find the latest time directory in the case (largest float-parsed name).
// Used by dmrRestart to pass an explicit -time to decomposePar so that
// moving-mesh cases correctly re-decompose <time>/polyMesh/points instead
// of the (-latestTime + -copyZero) path that copies serial points verbatim.
// Returns the directory basename (e.g. "0.085"), or empty string if none
// found — the caller is expected to abort in that case.
static std::string dmrLatestTimeName(const std::string& casePath)
{
    // Glob direct children whose names start with a digit.  This matches
    // OpenFOAM time names ("0", "0.05", "1.5", "100", ...) and excludes
    // "constant", "system", "processor*", "log.dmr", etc.
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
    // Determine before writing whether this is a DMR-forced checkpoint (i.e.
    // not a natural writeInterval boundary).  We mark forced dirs with a
    // sentinel file so they can be deleted once a newer checkpoint exists —
    // they are DMR artefacts, not meaningful simulation output.
    const bool isDmrForced = !runTime.writeTime();

    // All ranks flush their fields to disk before rank 0 reconstructs.
    // We deliberately do NOT gate on the return value of writeNow():
    // Time::write() returns the AND-reduction over every registered
    // regIOobject's writeObject() success, including transient
    // dynamic-mesh sub-objects (topo-changer caches, AMI weights,
    // fvMeshStitcher state) that can legitimately return false at an
    // arbitrary step without any data loss.  The bool is therefore not
    // a meaningful "did the checkpoint succeed" signal.  Real I/O
    // failures (disk full, permissions, stream errors) propagate via
    // OpenFOAM's FatalError mechanism, which terminates the run before
    // we reach the reconstruct subprocess below.
    runTime.writeNow();

    // Wait for all parallel writes to land before rank 0 opens them.
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

        // reconstructPar flags:
        //   -newTimes   merge every time dir present in processors but not
        //               yet in serial.  These are the user's writeInterval
        //               checkpoints (scientific output) and DMR-forced
        //               checkpoints; both must be reconstructed to serial
        //               so the user keeps them and the new process group
        //               can restart from latestTime.
        //   -allRegions multi-region for foamMultiRun.
        //
        // -rm is deliberately omitted.  On dynamic mesh, removing
        // processor*/<earlier>/ after each successful merge regresses the
        // procMesh's facesInstance back to constant/ while
        // completeMesh.facesInstance() stays at the just-merged time —
        // the next iteration of the walk then trips
        // compareInstances == -1 in
        // domainDecomposition::readUpdateReconstruct
        // (domainDecomposition.C:899) and aborts with
        // "complete mesh topology has evolved further than the processor
        // mesh topology."  We sweep processor*/<time>/ ourselves below,
        // after reconstructPar has exited and is no longer walking.
        std::string cmd =
            "reconstructPar -newTimes -case '" + casePath + "'";
        if (allRegions) cmd += " -allRegions";

        dmrRunOrAbort(casePath, "dmr_reconstruct.log", cmd);

        // Post-walk cleanup: every <time>/ now in serial form has its
        // processor*/<time>/ counterpart removed.  reconstructPar has
        // exited so there is no walking instance resolver left to trip.
        // We only remove a processor*/<time>/ if the matching serial
        // <time>/ exists — guards against accidentally deleting a
        // processor copy whose serial reconstruct silently no-op'd.
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

        // Cheap insurance against filesystem coherence quirks before the
        // new process group's decomposePar reads what we just wrote.  On
        // POSIX-coherent filesystems (GPFS, Lustre, local ext4) this is a
        // no-op for our case.  On Docker bind mounts we observed write
        // visibility delays — sync() forces buffered writes to be flushed.
        ::sync();

        dmrLifecycleLog(casePath, "Reconstruction complete.");

        // ---- Sentinel-based checkpoint cleanup ----------------------------
        //
        // SAFETY NOTE on deleting the previous sentinel: the data at the
        // older sentinel time was already loaded into the new process
        // group's memory at the previous reconfig, the simulation has
        // advanced past it (in memory) before this reconfig fires, and
        // the just-touched newer sentinel is what restart-from-disk would
        // pick up.  So the older dir is no longer a useful restart anchor;
        // removing it from disk does not lose simulation history.
        //
        // OF time names are decimals (e.g. "0.085", "0.17") and must be
        // compared as floats — version sort would order "0.17" before
        // "0.085" (since version sort segments at '.' and compares each
        // segment as an integer: 17 < 85).
        if (isDmrForced)
        {
            const std::string sentinel =
                casePath + "/" + timeName + "/.dmr_checkpoint";
            if (FILE* s = std::fopen(sentinel.c_str(), "w"))
                std::fclose(s);

            // Glob every sentinel-bearing time dir, parse time as double,
            // sort, and remove all but the newest.  Foam::rmDir is the
            // OpenFOAM-internal recursive remove (POSIX-backed), avoiding
            // a subprocess fork per cleanup.
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
                    catch (const std::exception&)
                    {
                        // Non-numeric dir name — skip rather than crash.
                    }
                }
                globfree(&g);

                std::sort(dirs.begin(), dirs.end());

                // Drop the newest (last after ascending sort), remove rest.
                for (size_t i = 0; i + 1 < dirs.size(); ++i)
                {
                    Foam::rmDir(dirs[i].second);
                }
            }
        }
    }

    // Barrier: keep old-group ranks in lockstep before they exit DMR_AUTO
    // and proceed into MPI_Finalize via dmr_reconfigure().
    MPI_Barrier(MPI_COMM_WORLD);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void dmrRestart(const std::string& casePath, bool allRegions)
{
    // Called by the NEW process group inside DMR_AUTO.  The child processes
    // spawned here run in isolated MPI environments, so there is no
    // interaction with DMR_INTERCOMM — a single-phase restart is safe.

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

        // ---- Update decomposeParDict ---------------------------------------

        // numberOfSubdomains must match the new process count.
        dmrUpdateDictEntry
        (
            casePath, "system/decomposeParDict",
            "numberOfSubdomains", std::to_string(newSize)
        );

        // Force scotch: geometric decomposers (hierarchical, simple)
        // require n-coefficients that do not hold when DMR changes the
        // process count.
        dmrUpdateDictEntry
        (
            casePath, "system/decomposeParDict",
            "decomposer", "scotch"
        );

        // ---- Update controlDict --------------------------------------------

        dmrUpdateDictEntry
        (
            casePath, "system/controlDict",
            "startFrom", "latestTime"
        );

        // TODO multi-region: foamMultiRun cases may have per-region
        // system/<region>/decomposeParDict and system/<region>/controlDict.
        // For -allRegions decomposition the root system/decomposeParDict is
        // what controls subdomain count, so the above is sufficient for the
        // cases tested in FOAMMULTIRUN_TEST_FINDINGS.md.  Per-region
        // overrides not handled here.

        // ---- Decompose serial case for the new process count ---------------
        //
        // decomposePar flags:
        //   -force      remove any pre-existing processor* before decomposing
        //   -time <T>   decompose specifically at time T.  Pass an explicit
        //               time rather than -latestTime because moving-mesh
        //               cases (constant/dynamicMeshDict, <time>/polyMesh/
        //               points written) need points decomposed not copied.
        //               -latestTime with default behaviour falls back to the
        //               serial points → all processors get the full point
        //               set → face area mismatch at processor boundaries
        //               → crash.  See FOAMMULTIRUN_TEST_FINDINGS.md.  -time
        //               is universally correct for static and dynamic mesh.
        //   -cellProc   required for NCC cyclic patches (nFaces is 0 in
        //               serial, so cellProc must be regenerated)
        //   -allRegions multi-region for foamMultiRun

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

        // For dynamic-mesh cases, OF only writes a `points` file at later
        // times when only the points moved (no topology change).  Copy
        // the missing topology files (boundary, faces, owner, neighbour,
        // ...) from the most recent earlier time that has them, so
        // decomposePar -time <restartTime> finds a complete polyMesh.
        // No-op for static cases.
        dmrEnsureFullPolyMesh(casePath, restartTime, allRegions);

        std::string cmd =
            "decomposePar -force -time " + restartTime + " -cellProc"
          + " -case '" + casePath + "'";
        if (allRegions) cmd += " -allRegions";

        dmrRunOrAbort(casePath, "dmr_decompose.log", cmd);

        // Engine cases (e.g. tutorials/CHT/engine2Valve2D) keep extra
        // sub-meshes under constant/meshes/<name> that decomposePar
        // -allRegions does not visit.  Without re-decomposing them for
        // the new process count, the cyclicAMI weights on the second
        // reconfiguration end up being computed against a stale layout
        // and decomposePar crashes inside fvMeshStitcher::connectThis
        // with a 0/0 surface-interpolation weight.  Mirror the pattern
        // used by the case's own Allrun: after the main decomposePar,
        // loop over constant/meshes/* and decompose each sub-mesh
        // explicitly.  No-op for cases that do not have sub-meshes.
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

    // All NEW-group ranks synchronise before proceeding to setRootCase.H,
    // which re-initialises OpenFOAM in parallel mode on the new world.
    MPI_Barrier(MPI_COMM_WORLD);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void dmrFinalize(const std::string& casePath, bool allRegions)
{
    // Called after dmr_finalize().  Merge any remaining processor dirs
    // into the serial case, sweep DMR-forced sentinels, and remove the
    // processor* directories now that the run is complete.

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

            // ---- processor* cleanup ---------------------------------------
            // Only delete on successful reconstruction.  After a clean
            // finish the case is ready for serial post-processing and
            // processor* dirs are dead weight.  On failure we keep them
            // so the user can debug the parallel state.
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

        // Remove all remaining DMR-forced sentinel dirs.  These are
        // superseded by whatever the time loop wrote at or before endTime
        // (or by the final reconstructPar above).
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

void dmrEnsureFullPolyMesh
(
    const std::string& casePath,
    const std::string& restartTime,
    bool allRegions
)
{
    if (!dmrHasDynamicMesh(casePath)) return;

    // Files OF requires for a complete polyMesh at decomposition time.
    // points is omitted because that is the file the dynamic-mesh writer
    // emits at every changed-points time; it is what we want to keep.
    static const std::vector<std::string> topologyFiles =
        {"boundary", "faces", "owner", "neighbour", "faceZones"};

    // Build the list of polyMesh dirs to inspect.
    // - allRegions=false  → <case>/<time>/polyMesh
    // - allRegions=true   → <case>/<time>/<region>/polyMesh for each
    //                       region present at <time>
    std::vector<std::string> polyMeshDirs;
    if (!allRegions)
    {
        polyMeshDirs.push_back(casePath + "/" + restartTime + "/polyMesh");
    }
    else
    {
        const std::string regionGlob =
            casePath + "/" + restartTime + "/*/polyMesh";
        glob_t rg;
        if (glob(regionGlob.c_str(), GLOB_ONLYDIR, nullptr, &rg) == 0)
        {
            for (size_t i = 0; i < rg.gl_pathc; ++i)
            {
                polyMeshDirs.emplace_back(rg.gl_pathv[i]);
            }
            globfree(&rg);
        }
    }

    if (polyMeshDirs.empty()) return;

    // Sorted list of all time directory names (ascending), used to walk
    // backwards from restartTime when topology files are missing.
    glob_t tg;
    const std::string timeGlob = casePath + "/[0-9]*";
    std::vector<std::pair<double, std::string>> times;
    if (glob(timeGlob.c_str(), GLOB_ONLYDIR, nullptr, &tg) == 0)
    {
        for (size_t i = 0; i < tg.gl_pathc; ++i)
        {
            std::string p(tg.gl_pathv[i]);
            const auto slash = p.find_last_of('/');
            const std::string name = p.substr(slash + 1);
            try { times.emplace_back(std::stod(name), name); }
            catch (const std::exception&) {}
        }
        globfree(&tg);
    }
    std::sort(times.begin(), times.end());

    double restartT = -1.0;
    try { restartT = std::stod(restartTime); } catch (const std::exception&) {}

    for (const auto& pmDir : polyMeshDirs)
    {
        // Region tag derived from the polyMesh path (e.g. ".../1/fluid/
        // polyMesh" → region = "fluid"; ".../1/polyMesh" → region = "").
        const std::string suffix = "/polyMesh";
        const std::string parent =
            pmDir.substr(0, pmDir.size() - suffix.size());
        const std::string parentLeaf =
            parent.substr(parent.find_last_of('/') + 1);
        const bool perRegion = (parentLeaf != restartTime);
        const std::string region = perRegion ? parentLeaf : std::string();

        for (const auto& f : topologyFiles)
        {
            if (Foam::isFile(pmDir + "/" + f)) continue;

            // Walk back through earlier times to find this file.
            for (auto it = times.rbegin(); it != times.rend(); ++it)
            {
                if (it->first >= restartT) continue;
                const std::string candidate =
                    perRegion
                  ? casePath + "/" + it->second + "/" + region
                          + "/polyMesh/" + f
                  : casePath + "/" + it->second + "/polyMesh/" + f;
                if (Foam::isFile(candidate))
                {
                    const std::string cp =
                        "cp '" + candidate + "' '" + pmDir + "/'";
                    std::system(cp.c_str());
                    break;
                }
            }
        }
    }
}


bool dmrHasDynamicMesh(const std::string& casePath)
{
    // Single-region case
    if (Foam::isFile(casePath + "/constant/dynamicMeshDict"))
    {
        return true;
    }

    // Multi-region case: any constant/<region>/dynamicMeshDict
    const std::string pattern = casePath + "/constant/*/dynamicMeshDict";
    glob_t g;
    bool found = false;
    if (glob(pattern.c_str(), 0, nullptr, &g) == 0)
    {
        found = g.gl_pathc > 0;
        globfree(&g);
    }
    return found;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
