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
#include "Pstream.H"

#include <dmr.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <string>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Run an external command, stream its output to <case>/logs/<logName>,
// and abort the whole job on non-zero exit.  Only called by master rank.
static void dmrRunOrAbort
(
    const std::string& casePath,
    const std::string& logName,
    const std::string& utilityArgs
)
{
    const std::string logPath = casePath + "/logs/" + logName;

    const std::string cmd =
        "mkdir -p '" + casePath + "/logs' && "
      + utilityArgs + " > '" + logPath + "' 2>&1";

    const int ret = std::system(cmd.c_str());

    if (ret != 0)
    {
        std::fprintf
        (
            stderr,
            "DMR: FATAL — '%s' exited with status %d; see '%s'.\n",
            utilityArgs.c_str(), ret, logPath.c_str()
        );
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


// Update a single entry in a case dictionary via foamDictionary.
// Aborts the job on failure — the restart cannot proceed with stale dicts.
static void dmrSetDictEntry
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

    dmrRunOrAbort(casePath, "dmr_foamDictionary.log", cmd);
}


// Append one row to <case>/logs/dmr_analytics.csv.
// Writes the CSV header on first call (file position 0).
// Only called by master rank.
static void dmrWriteAnalytics
(
    const std::string& casePath,
    const DMRAnalytics& a,
    const char* phase
)
{
    const std::string csvPath = casePath + "/logs/dmr_analytics.csv";

    FILE* f = std::fopen(csvPath.c_str(), "a");
    if (!f)
    {
        std::fprintf
        (
            stderr,
            "DMR: warning — cannot open '%s' for analytics logging.\n",
            csvPath.c_str()
        );
        return;
    }

    // Write header only when the file is newly created (append position == 0).
    if (std::ftell(f) == 0)
    {
        std::fprintf
        (
            f,
            "phase,event_time,event,world_size,node_count,"
            "reconfiguration_time_s,communication_efficiency,pending_nodes\n"
        );
    }

    std::fprintf
    (
        f,
        "%s,%.3f,%s,%d,%d,%.4f,%.4f,%d\n",
        phase,
        a.event_time,
        dmr_get_analytics_event_str(a.event),
        a.world_size,
        a.node_count,
        a.reconfiguration_time,
        a.communication_efficiency,
        a.pending_nodes
    );

    std::fclose(f);
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

        Info<< "DMR: Reconstructing processor directories..." << nl;

        // reconstructPar flags:
        //   -newTimes   only merge time dirs not already present in serial
        //   -rm         remove processor*/<time> after successful merge
        //   -allRegions multi-region for foamMultiRun
        std::string cmd =
            "reconstructPar -newTimes -rm -case '" + casePath + "'";
        if (allRegions) cmd += " -allRegions";

        dmrRunOrAbort(casePath, "dmr_reconstruct.log", cmd);

        Info<< "DMR: Reconstruction complete." << nl;

        // ---- Sentinel-based checkpoint cleanup ----------------------------
        // Touch a hidden marker in the reconstructed time dir.  Then sweep
        // out any older marker dirs, keeping only this (newest) one on disk.
        // sort -V: version-style sort (0.1 < 1.0 < 10.0) — correct for OF
        // time names.  head -n -1: drop the last (newest) line.
        // xargs -r: skip execution entirely when stdin is empty (GNU xargs).
        if (isDmrForced)
        {
            const std::string sentinel =
                casePath + "/" + timeName + "/.dmr_checkpoint";
            if (FILE* s = std::fopen(sentinel.c_str(), "w"))
                std::fclose(s);

            std::system
            (
                ("find '" + casePath + "' -maxdepth 2"
                 " -name '.dmr_checkpoint' -printf '%h\\n'"
                 " | sort -V | head -n -1"
                 " | xargs -r -I{} rm -rf '{}'").c_str()
            );
        }

        // ---- Analytics ---------------------------------------------------
        // dmr_get_analytics() fills per-reconfiguration metrics that are
        // only available on rank 0.  Log to CSV for the overhead analysis.
        DMRAnalytics analytics;
        if (dmr_get_analytics(&analytics) == DMR_SUCCESS)
        {
            dmrWriteAnalytics(casePath, analytics, "checkpoint");
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
        std::fprintf
        (
            stderr,
            "DMR: Restarting with %d processes.\n",
            newSize
        );

        // ---- Update decomposeParDict ----------------------------------------

        // numberOfSubdomains must match the new process count.
        dmrSetDictEntry
        (
            casePath,
            "system/decomposeParDict",
            "numberOfSubdomains",
            std::to_string(newSize)
        );

        // Force scotch: geometric decomposers (hierarchical, simple) require
        // n-coefficients that do not hold when DMR changes the process count.
        dmrSetDictEntry
        (
            casePath,
            "system/decomposeParDict",
            "decomposer",
            "scotch"
        );

        // ---- Update controlDict --------------------------------------------

        dmrSetDictEntry
        (
            casePath,
            "system/controlDict",
            "startFrom",
            "latestTime"
        );

        // ---- Decompose serial case for the new process count ---------------

        // decomposePar flags:
        //   -force       remove any pre-existing processor* before decomposing
        //   -latestTime  decompose only the restart time (picks up moved
        //                polyMesh/points for dynamic mesh cases)
        //   -cellProc    required for NCC cyclic patches (nFaces is 0 in
        //                serial, so cellProc must be regenerated)
        //   -allRegions  multi-region for foamMultiRun
        std::string cmd =
            "decomposePar -force -latestTime -cellProc"
            " -case '" + casePath + "'";
        if (allRegions) cmd += " -allRegions";

        dmrRunOrAbort(casePath, "dmr_decompose.log", cmd);

        std::fprintf(stderr, "DMR: Decomposition complete.\n");

        // ---- Analytics ---------------------------------------------------
        DMRAnalytics analytics;
        if (dmr_get_analytics(&analytics) == DMR_SUCCESS)
        {
            dmrWriteAnalytics(casePath, analytics, "restart");
        }
    }

    // All NEW-group ranks synchronise before proceeding to setRootCase.H,
    // which re-initialises OpenFOAM in parallel mode on the new world.
    MPI_Barrier(MPI_COMM_WORLD);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void dmrFinalize(const std::string& casePath, bool allRegions)
{
    // Called after dmr_finalize() — DMR is no longer active at this point,
    // so no analytics call here.  Merge any remaining processor dirs into
    // the serial case, then sweep all DMR-forced checkpoint dirs that were
    // never superseded by a natural write.

    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        Info<< "DMR: Final reconstruction to serial..." << nl;

        std::string cmd =
            "reconstructPar -newTimes -rm -case '" + casePath + "'";
        if (allRegions) cmd += " -allRegions";

        const std::string logPath = casePath + "/logs/dmr_final.log";
        const std::string fullCmd =
            "mkdir -p '" + casePath + "/logs' && "
          + cmd + " > '" + logPath + "' 2>&1";

        const int ret = std::system(fullCmd.c_str());

        if (ret != 0)
        {
            WarningInFunction
                << "DMR: final reconstructPar failed (exit " << ret
                << "). Processor directories may remain; see "
                << logPath.c_str() << nl;
        }
        else
        {
            Info<< "DMR: Final reconstruction complete." << nl;
        }

        // Remove all remaining DMR-forced checkpoint dirs.  These are
        // superseded by whatever the time loop wrote at or before endTime.
        std::system
        (
            ("find '" + casePath + "' -maxdepth 2"
             " -name '.dmr_checkpoint' -printf '%h\\n'"
             " | xargs -r -I{} rm -rf '{}'").c_str()
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
