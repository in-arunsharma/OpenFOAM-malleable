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

Application
    dmrRelayout

Description
    Writes the output of a malleable (DMR) run in a single, consistent
    decomposition.

    A malleable run changes its process count while it runs, so its natural
    writes land in whatever layout it happens to be using at the time: the
    time series ends up decomposed N ways at one write and N' ways at the
    next, which is not readable as a decomposed case. This utility receives
    the solver's state and re-emits it in the layout the run is always able
    to produce, MIN = policy-min nodes times procs-per-node, so every output
    time shares one decomposition.

    It is not run by hand. The solver spawns it with MPI_Comm_spawn at a
    natural write, with the same number of ranks as the solver itself, and
    hands over the mesh and fields rank-for-rank over the parent
    intercommunicator; nothing passes through disk. The solver then carries
    on immediately while this utility redistributes to the MIN layout,
    writes, and exits, freeing its memory.

    The equal world sizes are required rather than incidental:
    fvMeshDistribute runs on max(N_src, N_dst) ranks and MIN <= N, so the
    handoff is a straight 1:1 copy with no re-decomposition in flight, and
    the redistribution to MIN happens entirely inside this world.

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "dmrRedist.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "setRootCaseNoFunctionObjects.H"
    #include "createTime.H"

    dmrRelayoutFromParent(runTime);

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
