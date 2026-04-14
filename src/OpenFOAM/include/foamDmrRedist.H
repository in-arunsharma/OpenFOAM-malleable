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

Description
    DMR redistribution helpers shared by foamRun and foamMultiRun.

    dmrCheckpoint() — called by OLD processes before they exit.
        Writes fields, reconstructs to serial, removes processor dirs.

    dmrRestart() — called by NEW processes at startup.
        Updates decomposeParDict, decomposes for new process count.

    Both use system() to invoke existing OpenFOAM utilities as external
    processes.  Only rank 0 runs the commands; other ranks wait at a
    barrier.

    Key fix vs v1: uses "decomposePar -time T -cellProc" instead of
    "-latestTime" to correctly handle moving-mesh cases where
    <time>/polyMesh/points must be decomposed (not just copied).

\*---------------------------------------------------------------------------*/

#ifndef foamDmrRedist_H
#define foamDmrRedist_H

#include "Pstream.H"
#include "Time.H"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <mpi.h>
#include <dirent.h>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Find the latest time directory in a case by scanning the filesystem.
// Runs before OpenFOAM is initialised, so must not use Foam:: types.
inline std::string dmrFindLatestTime(const std::string& casePath)
{
    DIR* dir = opendir(casePath.c_str());
    if (!dir) return "0";

    double maxTime = -1;
    std::string maxTimeName;
    struct dirent* entry;

    while ((entry = readdir(dir)) != nullptr)
    {
        char* endptr;
        double val = strtod(entry->d_name, &endptr);

        // Accept only pure numeric names (e.g. "0", "0.001", "1.5")
        if (*endptr == '\0' && endptr != entry->d_name)
        {
            if (val > maxTime)
            {
                maxTime = val;
                maxTimeName = entry->d_name;
            }
        }
    }

    closedir(dir);
    return maxTimeName.empty() ? "0" : maxTimeName;
}


// Called by OLD processes before exiting on a DMR reconfiguration.
// Writes all fields, reconstructs to serial, and removes processor dirs.
inline void dmrCheckpoint(Foam::Time& runTime, bool allRegions = false)
{
    using namespace Foam;

    // All ranks write their fields to disk
    runTime.writeNow();

    // Ensure all writes are on disk before rank 0 reconstructs
    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        const fileName casePath = runTime.globalPath();

        Info<< "DMR: Reconstructing to serial..." << nl;

        // -newTimes: reconstruct all unprocessed times (efficient)
        // -rm:       remove processor dirs after reconstruction
        std::string cmd("reconstructPar -case ");
        cmd += casePath;
        cmd += " -newTimes -rm";
        if (allRegions) cmd += " -allRegions";

        const int ret = ::system(cmd.c_str());

        if (ret != 0)
        {
            FatalErrorInFunction
                << "reconstructPar failed with exit code " << ret
                << exit(FatalError);
        }

        Info<< "DMR: Reconstruction complete." << nl;
    }

    // Wait for rank 0 to finish before all old procs exit
    MPI_Barrier(MPI_COMM_WORLD);
}


// Called by NEW processes at startup after a DMR reconfiguration.
// Runs BEFORE OpenFOAM is initialised (before setRootCase.H), so
// must not use Foam::Info, Pstream, FatalError, or fileName.
inline void dmrRestart(const std::string& casePath, bool allRegions = false)
{
    int newSize, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &newSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    if (myRank == 0)
    {
        std::fprintf(stderr, "DMR: Restarting with %d processes.\n", newSize);

        // --- Update numberOfSubdomains in decomposeParDict ---
        const std::string dictPath =
            casePath + "/system/decomposeParDict";

        std::string sedCmd =
            "sed -i 's/numberOfSubdomains[[:space:]].*/"
            "numberOfSubdomains  " + std::to_string(newSize) + ";/' "
            + dictPath;

        int ret = ::system(sedCmd.c_str());

        if (ret != 0)
        {
            std::fprintf(stderr,
                "DMR: FATAL — failed to update decomposeParDict "
                "(exit code %d)\n", ret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // --- Force scotch decomposer ---
        // Geometric decomposers (hierarchical, simple) require matching
        // n-coefficients that break when DMR changes the count.
        const std::string sedDecomp =
            "sed -i 's/decomposer[[:space:]].*/"
            "decomposer      scotch;/' "
            + dictPath;

        ret = ::system(sedDecomp.c_str());

        if (ret != 0)
        {
            std::fprintf(stderr,
                "DMR: WARNING — failed to force scotch decomposer "
                "(exit code %d)\n", ret);
        }

        // --- Ensure startFrom is set to latestTime ---
        const std::string ctrlDictPath =
            casePath + "/system/controlDict";

        const std::string sedStart =
            "sed -i 's/startFrom[[:space:]].*/"
            "startFrom       latestTime;/' "
            + ctrlDictPath;

        ret = ::system(sedStart.c_str());

        if (ret != 0)
        {
            std::fprintf(stderr,
                "DMR: WARNING — failed to set startFrom "
                "(exit code %d)\n", ret);
        }

        // --- Decompose for new process count ---
        // Use -time T (not -latestTime) to correctly decompose
        // moved polyMesh/points in dynamic-mesh cases.
        // Use -cellProc for correct processor assignment.
        const std::string restartTime = dmrFindLatestTime(casePath);

        std::fprintf(stderr,
            "DMR: Decomposing for %d procs (time=%s)...\n",
            newSize, restartTime.c_str());

        std::string decompCmd("decomposePar -case ");
        decompCmd += casePath;
        decompCmd += " -force -time " + restartTime + " -cellProc";
        if (allRegions) decompCmd += " -allRegions";

        ret = ::system(decompCmd.c_str());

        if (ret != 0)
        {
            std::fprintf(stderr,
                "DMR: FATAL — decomposePar failed "
                "(exit code %d)\n", ret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::fprintf(stderr, "DMR: Decomposition complete.\n");
    }

    // Wait for rank 0 to finish before all new procs continue
    MPI_Barrier(MPI_COMM_WORLD);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
