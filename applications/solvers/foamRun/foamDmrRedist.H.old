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
    DMR redistribution helpers for foamRun.

    dmrCheckpoint() is called by OLD processes before they exit.
    It writes fields, reconstructs to serial, and removes processor dirs.

    dmrRestart() is called by NEW processes at startup.
    It gets the new process count, updates decomposeParDict, and decomposes.

    Both use system() to invoke existing OpenFOAM utilities (reconstructPar,
    decomposePar) as external processes. Only rank 0 runs the commands;
    other ranks wait at a barrier.

\*---------------------------------------------------------------------------*/

#ifndef foamDmrRedist_H
#define foamDmrRedist_H

#include "Pstream.H"
#include "Time.H"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <mpi.h>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

inline void dmrCheckpoint(Foam::Time& runTime)
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

        // Use -newTimes so ALL intermediate writes (writeInterval)
        // are reconstructed to serial, not just the latest.
        // -rm then removes processor dirs.  Previous checkpoint
        // times already in serial are skipped (efficient).
        const std::string reconstructCmd =
            "reconstructPar -case " + casePath + " -newTimes -rm";

        const int ret = ::system(reconstructCmd.c_str());
        
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


inline void dmrRestart(const std::string& casePath)
{
    // NOTE: This runs BEFORE OpenFOAM is initialized (before setRootCase.H),
    // so we cannot use Foam::Info, Pstream, FatalError, or fileName.
    // Use raw MPI + C stdio instead.

    int newSize, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &newSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    if (myRank == 0)
    {
        std::fprintf(stderr, "DMR: Restarting with %d processes.\n", newSize);

        // Update numberOfSubdomains in decomposeParDict
        const std::string dictPath = casePath + "/system/decomposeParDict";

        const std::string sedCmd =
            "sed -i 's/numberOfSubdomains[[:space:]].*/"
            "numberOfSubdomains  " + std::to_string(newSize) + ";/' "
            + dictPath;

        int ret = ::system(sedCmd.c_str());

        if (ret != 0)
        {
            std::fprintf(stderr, "DMR: FATAL — failed to update "
                "decomposeParDict (exit code %d)\n", ret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Force scotch decomposer so any numberOfSubdomains works.
        // Geometric decomposers (hierarchical, simple) require matching
        // n-coefficients that would break when DMR changes the count.
        const std::string sedDecomp =
            "sed -i 's/decomposer[[:space:]].*/"
            "decomposer      scotch;/' "
            + dictPath;

        ret = ::system(sedDecomp.c_str());

        if (ret != 0)
        {
            std::fprintf(stderr, "DMR: WARNING — failed to update "
                "decomposer to scotch (exit code %d)\n", ret);
        }

        // Ensure startFrom is set to latestTime so the restarted solver
        // picks up from the checkpoint, not from time 0.
        const std::string ctrlDictPath = casePath + "/system/controlDict";

        const std::string sedStartFrom =
            "sed -i 's/startFrom[[:space:]].*/"
            "startFrom       latestTime;/' "
            + ctrlDictPath;

        ret = ::system(sedStartFrom.c_str());

        if (ret != 0)
        {
            std::fprintf(stderr, "DMR: WARNING — failed to update "
                "startFrom in controlDict (exit code %d)\n", ret);
        }

        std::fprintf(stderr, "DMR: Decomposing for %d processes...\n",
            newSize);

        const std::string decomposeCmd =
            "decomposePar -case " + casePath + " -force -latestTime";

        ret = ::system(decomposeCmd.c_str());

        if (ret != 0)
        {
            std::fprintf(stderr, "DMR: FATAL — decomposePar failed "
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
