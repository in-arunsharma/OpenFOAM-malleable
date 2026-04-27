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
    Checkpoint/restart helpers for DMR-driven malleable solvers.

    On reconfiguration, the OLD process group calls dmrCheckpoint() to
    flush its fields and reconstruct them into a serial case on disk;
    the NEW group calls dmrRestart() to update the case dictionaries and
    decompose for the new process count.  After the time loop ends,
    dmrFinalize() merges any leftover processor directories and removes
    them.

    reconstructPar and decomposePar are invoked as subprocesses (the
    only path that works across MPI sessions; in-process reuse fails
    with MPI_ERR_TRUNCATE on the post-restart communicator).  All other
    operations (dictionary edits, directory removal, sentinel handling)
    are performed in-process via Foam::dictionary, Foam::rmDir and the
    POSIX glob API.

    Implemented in dmrRedist.C and compiled into libfoamDmr, linked by
    foamRun and foamMultiRun.

\*---------------------------------------------------------------------------*/

#ifndef dmrRedist_H
#define dmrRedist_H

#include "Time.H"

#include <string>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

//- Called by OLD processes on a DMR reconfiguration.
//  Writes the current time via runTime.writeNow(), then invokes
//  reconstructPar as a subprocess to merge processor directories into the
//  serial case and remove them.  On subprocess failure the job is aborted
//  via MPI_Abort to prevent the NEW process group from reading a partially
//  reconstructed case.
//  \param runTime     The solver's current Time object (already initialised).
//  \param allRegions  True for foamMultiRun multi-region cases.
void dmrCheckpoint(Time& runTime, bool allRegions = false);

//- Called by NEW processes at startup (inside DMR_AUTO).
//  Updates system/decomposeParDict (numberOfSubdomains, decomposer) and
//  system/controlDict (startFrom latestTime) in process via Foam::dictionary,
//  then decomposes the latest serial time directory for the new process
//  count via a decomposePar subprocess.  decomposePar is invoked with
//  -time <latest> rather than -latestTime so that moving-mesh cases
//  re-decompose <time>/polyMesh/points correctly (see
//  FOAMMULTIRUN_TEST_FINDINGS.md).
//  \param casePath    Absolute path to the case directory.
//  \param allRegions  True for foamMultiRun multi-region cases.
void dmrRestart(const std::string& casePath, bool allRegions = false);

//- Called after dmr_finalize() to merge remaining processor directories
//  into the serial case and remove all DMR-forced checkpoint sentinels.
//  Uses WarningInFunction (not MPI_Abort) on reconstruction failure —
//  the simulation has already finished, so the case state is still valid.
//  \param casePath    Absolute path to the case directory.
//  \param allRegions  True for foamMultiRun multi-region cases.
void dmrFinalize(const std::string& casePath, bool allRegions = false);

} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
