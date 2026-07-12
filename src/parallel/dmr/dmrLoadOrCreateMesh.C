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
    Compiles the upstream loadOrCreateMesh translation unit (kept by
    openfoam.org inside the redistributePar utility) into libfoamDmr,
    without moving or duplicating the source. The grow-side in-process
    redistribute needs it to fabricate zero-cell meshes on ranks that
    have no processor data yet.

    Single source of truth: the utility's own file, included textually
    via the $(FOAM_UTILITIES) include path in Make/options. If upstream
    relocates or changes it, this build breaks loudly rather than
    diverging silently. The clean long-term home is a library (ESI
    ships it as fvMeshTools::loadOrCreateMesh); proposing that
    relocation upstream is tracked in the project's issue drafts and is
    the maintainers' call, not ours.

\*---------------------------------------------------------------------------*/

#include "loadOrCreateMesh.C"

// ************************************************************************* //
