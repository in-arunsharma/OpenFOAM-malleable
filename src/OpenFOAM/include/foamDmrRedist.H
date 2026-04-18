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
    Forwarding header — pulls in the libfoamDmr checkpoint/restart API.

    dmrCheckpoint() — called by OLD processes on a DMR reconfiguration.
        Writes all fields, reconstructs to a serial case on disk, and
        removes the processor time directories.

    dmrRestart() — called by NEW processes at startup.
        Updates decomposeParDict (new processor count, scotch decomposer)
        and controlDict (startFrom latestTime), then decomposes the serial
        case for the new process count.

    Both are implemented in src/parallel/dmr/dmrRedist.C and compiled
    into libfoamDmr, which is linked by foamRun and foamMultiRun.

\*---------------------------------------------------------------------------*/

#ifndef foamDmrRedist_H
#define foamDmrRedist_H

#include "dmrRedist.H"

#endif

// ************************************************************************* //
