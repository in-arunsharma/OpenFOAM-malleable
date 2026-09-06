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
//  in which case the write is redirected to dmrUniformWriteInPlace,
//  which emits the MIN layout instead of the running one.
void dmrWrite(Time& runTime, bool allRegions = false);

//- Natural-write hook. A malleable run writes its output in whatever
//  layout it is running at, so the time series ends up decomposed
//  differently from one write to the next and is not readable as a
//  case. This redistributes each live mesh down to the MIN layout
//  (policy-min nodes x procs-per-node) and writes THAT, so every
//  output time shares one decomposition however the run was resized.
//
//  Runs inside the solver's own ranks, on an in-memory clone of each
//  live mesh. fvMeshDistribute needs max(source, target) ranks and the
//  source is the running world, so no additional processes are
//  required: a malleable application must never need more resources to
//  reconfigure than it currently holds. This replaces an earlier
//  MPI_Comm_spawn of a same-width helper world, which exhausted the
//  SysV segment limit (shmmni) once both worlds shared a node.
//
//  The clone is separate and non-registered, so the live mesh is not
//  redistributed and the time loop continues at the running width. No
//  reconfiguration is involved and the solver is not restarted: this
//  is a natural checkpoint, not the forced one DMR takes before
//  resizing. No-op unless FOAM_DMR_UNIFORM_WRITE is unset or non-zero,
//  or if the run is already at the MIN layout.
void dmrUniformWriteInPlace(Time& runTime, bool allRegions = false);

//- Superseded by dmrUniformWriteInPlace; see the note there. Retained
//  because dmrRelayout links against it.
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
