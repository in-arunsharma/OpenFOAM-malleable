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
#include "IFstream.H"
#include "OFstream.H"
#include "dictionary.H"

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


// Replace one entry of an OpenFOAM case dictionary, in process.  Reads
// the dict from disk preserving its FoamFile header, calls dict.set()
// (which adds-or-overwrites), and writes it back.  Aborts on read/write
// failure — the restart cannot proceed with stale dicts.
//
// Replaces three foamDictionary subprocess spawns per reconfig.  The
// dictionary class auto-handles the FoamFile sub-entry because we read
// with keepHeader=true and write with subDict=false (i.e. without an
// outer { } wrapper since this is a top-level dict).
//
// The casePath is taken explicitly rather than inferred from filePath
// so the dict-update audit log can be written alongside the other DMR
// logs without string parsing.
template<class T>
static void dmrUpdateDictEntry
(
    const std::string& casePath,
    const std::string& filePath,
    const std::string& key,
    const T& value
)
{
    Foam::dictionary dict;

    {
        Foam::IFstream is(filePath);
        if (!is.good())
        {
            std::fprintf
            (
                stderr,
                "DMR: FATAL — cannot read '%s' for entry '%s'.\n",
                filePath.c_str(), key.c_str()
            );
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        dict.read(is, /* keepHeader = */ true);
    }

    dict.set(Foam::word(key), value);

    {
        Foam::OFstream os(filePath);
        if (!os.good())
        {
            std::fprintf
            (
                stderr,
                "DMR: FATAL — cannot write '%s' for entry '%s'.\n",
                filePath.c_str(), key.c_str()
            );
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        dict.write(os, /* subDict = */ false);
    }

    // Audit trail: one line per dict update in log.dmr/dmr_dict_updates.log.
    // Lets a reviewer see what dmrRestart modified between reconfigs
    // without having to diff the dicts.
    if (dmrLogEnabled())
    {
        Foam::mkDir(casePath + "/log.dmr");
        const std::string updatesLog =
            casePath + "/log.dmr/dmr_dict_updates.log";
        if (FILE* f = std::fopen(updatesLog.c_str(), "a"))
        {
            Foam::OStringStream oss;
            oss << value;
            std::fprintf
            (
                f,
                "%s  set  %s = %s\n",
                filePath.c_str(),
                key.c_str(),
                oss.str().c_str()
            );
            std::fclose(f);
        }
    }
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
    // Check the return value: Time::writeNow() returns false if any I/O in
    // the write path failed (disk full, permissions, truncated files), and
    // a partial write would silently corrupt the next restart.  We agree
    // across ranks via MPI_Allreduce(MIN) — a single rank failing is enough
    // to abort the whole job before master opens the processor dirs.
    const int writeOK = runTime.writeNow() ? 1 : 0;
    int writeOKAll = 0;
    MPI_Allreduce(&writeOK, &writeOKAll, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (!writeOKAll)
    {
        if (Pstream::master())
        {
            std::fprintf
            (
                stderr,
                "DMR: FATAL — runTime.writeNow() failed on at least one"
                " rank at t=%s; aborting to avoid a corrupt checkpoint.\n",
                runTime.name().c_str()
            );
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

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
        //   -newTimes   only merge time dirs not already present in serial
        //   -rm         remove processor*/<time> after successful merge
        //   -allRegions multi-region for foamMultiRun
        std::string cmd =
            "reconstructPar -newTimes -rm -case '" + casePath + "'";
        if (allRegions) cmd += " -allRegions";

        dmrRunOrAbort(casePath, "dmr_reconstruct.log", cmd);

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

        // ---- Update decomposeParDict ----------------------------------------
        //
        // In-process via Foam::dictionary IO — replaces three foamDictionary
        // subprocess spawns.  Faster (no fork+exec ×3) and surfaces parse
        // errors as proper OF FATAL ERRORs with line numbers, instead of
        // foamDictionary silently truncating the file (which is what we hit
        // when the case template had `location system;` unquoted).

        const std::string decomposeParDict =
            casePath + "/system/decomposeParDict";

        // numberOfSubdomains must match the new process count.
        dmrUpdateDictEntry
        (
            casePath, decomposeParDict,
            "numberOfSubdomains", label(newSize)
        );

        // Force scotch: geometric decomposers (hierarchical, simple)
        // require n-coefficients that do not hold when DMR changes the
        // process count.
        dmrUpdateDictEntry
        (
            casePath, decomposeParDict,
            "decomposer", word("scotch")
        );

        // ---- Update controlDict --------------------------------------------

        dmrUpdateDictEntry
        (
            casePath, casePath + "/system/controlDict",
            "startFrom", word("latestTime")
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

        std::string cmd =
            "decomposePar -force -time " + restartTime + " -cellProc"
          + " -case '" + casePath + "'";
        if (allRegions) cmd += " -allRegions";

        dmrRunOrAbort(casePath, "dmr_decompose.log", cmd);

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

} // End namespace Foam

// ************************************************************************* //
