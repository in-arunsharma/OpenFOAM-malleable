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
    Checkpoint / restart helpers for DMR-driven malleable solvers.

    On reconfiguration the OLD process group calls dmrCheckpoint() to
    flush its fields and reconstruct them to serial on disk; the NEW
    group calls dmrRestart() to update the case dictionaries and
    decompose for the new process count. dmrFinalize() runs after the
    time loop ends.

    reconstructPar and decomposePar are invoked as subprocesses;
    in-process reuse fails with MPI_ERR_TRUNCATE on the post-restart
    communicator. Compiled into libfoamDmr, linked by foamRun and
    foamMultiRun.

\*---------------------------------------------------------------------------*/

#ifndef dmrRedist_H
#define dmrRedist_H

#include "Time.H"

#include <string>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

//- OLD-group callback: writeNow + reconstructPar -newTimes -rm.
//  Aborts via MPI_Abort on subprocess failure to avoid handing a
//  partial state to the new group.
void dmrCheckpoint(Time& runTime, bool allRegions = false);

//- NEW-group callback (inside DMR_AUTO restart phase): update
//  decomposeParDict (numberOfSubdomains, decomposer) and controlDict
//  (startFrom latestTime), then decomposePar for the new process count.
//  For moving-mesh cases the -time argument is a comma-separated chain
//  starting from the most recent polyMesh-write time so decomposePar
//  evolves the topology forward correctly.
void dmrRestart(const std::string& casePath, bool allRegions = false);

//- NEW-group hook, included between createTime.H and createMesh.H
//  (dmrGrow.H). After a GROW reconfiguration the checkpoint exists
//  only in the old, smaller processor layout; this redistributes it
//  in-process to the current world size via fvMeshDistribute
//  (loadOrCreateMesh fabricates zero-cell meshes on the ranks that
//  have no data), replacing the serial decomposePar. No-op unless
//  this process generation follows a DMR GROW reconfiguration.
void dmrGrowRedistribute(Time& runTime, bool allRegions = false);

//- After the time loop ends: a final reconstructPar -newTimes -rm,
//  then sweep any remaining DMR-forced checkpoint sentinels.
//  Uses WarningInFunction (not MPI_Abort) on failure.
void dmrFinalize(const std::string& casePath, bool allRegions = false);

//- Replaces runTime.write() in the solver's time loop (dmrWrite.H).
//  Behaves exactly like it unless the uniform-output path is active,
//  in which case the fields are left out of this write and handed to a
//  spawned helper that re-emits them in the MIN layout instead.
void dmrWrite(Time& runTime, bool allRegions = false);

//- Natural-write hook. A malleable run writes its output in whatever
//  layout it is running at, so the time series ends up decomposed
//  differently from one write to the next and is not readable as a
//  case. This hands the state to a second MPI world spawned for the
//  purpose (same size, so the copy is 1:1 rank-to-rank and nothing
//  goes through disk), which redistributes it to the MIN layout
//  (policy-min nodes x procs-per-node), writes it, and exits while the
//  solver keeps running. No-op unless FOAM_DMR_UNIFORM_WRITE=1, or if
//  the run is already at the MIN layout.
void dmrSpawnUniformWrite(Time& runTime, bool allRegions = false);

//- Body of the spawned dmrRelayout helper: receive one rank's slice of
//  the mesh and fields from the parent world, redistribute to the MIN
//  layout, write it, and drop the directories of the ranks that fall
//  outside that layout. Aborts if not started via MPI_Comm_spawn.
void dmrRelayoutFromParent(Time& runTime);

} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
