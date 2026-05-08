/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2025 OpenFOAM Foundation
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
    Cell to face interpolation scheme. Included in fvMesh.

\*---------------------------------------------------------------------------*/

#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "demandDrivenData.H"
#include "coupledFvPatch.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(surfaceInterpolation, 0);
}


// * * * * * * * * * * * * Defensive degenerate-face guards * * * * * * * * //
//
// Each of the make...() functions below contains an unconditional VSMALL-
// threshold guard against zero-area / zero-distance faces that upstream
// OpenFOAM otherwise divides by, raising SIGFPE.  The guards are dormant
// for every real mesh face (where the geometric quantities are many
// orders of magnitude above VSMALL) and fire only on synthetic
// degenerate faces produced by fvMeshStitcher::connectThis at AMI patch
// pairs whose geometric overlap has collapsed to zero.  Such a face has
// zero physical contribution to any flux or gradient (its area appears
// multiplicatively in every surface integral), so the fallback values
// chosen are the unique values that produce zero contribution downstream:
//   - weight       -> 0    (interpolation weight on a zero-area face)
//   - deltaCoeff   -> GREAT (mimics 1/tiny_distance = huge in upstream;
//                            BCs that divide BY deltaCoeff get ~0)
//   - corrVec      -> Zero (zero correction vector)
// For valid meshes the upstream computation runs unchanged, so simulation
// results are bit-identical to unpatched OpenFOAM.


// * * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * //

void Foam::surfaceInterpolation::clearOut()
{
    deleteDemandDrivenData(weights_);
    deleteDemandDrivenData(deltaCoeffs_);
    deleteDemandDrivenData(nonOrthDeltaCoeffs_);
    deleteDemandDrivenData(nonOrthCorrectionVectors_);
}


// * * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * //

Foam::surfaceInterpolation::surfaceInterpolation(const fvMesh& fvm)
:
    mesh_(fvm),
    weights_(nullptr),
    deltaCoeffs_(nullptr),
    nonOrthDeltaCoeffs_(nullptr),
    nonOrthCorrectionVectors_(nullptr)
{}


// * * * * * * * * * * * * * * * * Destructor * * * * * * * * * * * * * * * //

Foam::surfaceInterpolation::~surfaceInterpolation()
{
    clearOut();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

const Foam::surfaceScalarField&
Foam::surfaceInterpolation::weights() const
{
    if (!weights_)
    {
        makeWeights();
    }

    return (*weights_);
}


const Foam::surfaceScalarField&
Foam::surfaceInterpolation::deltaCoeffs() const
{
    if (!deltaCoeffs_)
    {
        makeDeltaCoeffs();
    }

    return (*deltaCoeffs_);
}


const Foam::surfaceScalarField&
Foam::surfaceInterpolation::nonOrthDeltaCoeffs() const
{
    if (!nonOrthDeltaCoeffs_)
    {
        makeNonOrthDeltaCoeffs();
    }

    return (*nonOrthDeltaCoeffs_);
}


const Foam::surfaceVectorField&
Foam::surfaceInterpolation::nonOrthCorrectionVectors() const
{
    if (!nonOrthCorrectionVectors_)
    {
        makeNonOrthCorrectionVectors();
    }

    return (*nonOrthCorrectionVectors_);
}


bool Foam::surfaceInterpolation::movePoints()
{
    deleteDemandDrivenData(weights_);
    deleteDemandDrivenData(deltaCoeffs_);
    deleteDemandDrivenData(nonOrthDeltaCoeffs_);
    deleteDemandDrivenData(nonOrthCorrectionVectors_);

    return true;
}


void Foam::surfaceInterpolation::makeWeights() const
{
    if (debug)
    {
        Pout<< "surfaceInterpolation::makeWeights() : "
            << "Constructing weighting factors for face interpolation"
            << endl;
    }

    weights_ = new surfaceScalarField
    (
        IOobject
        (
            "weights",
            mesh_.pointsInstance(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false // Do not register
        ),
        mesh_,
        dimless
    );
    surfaceScalarField& weights = *weights_;

    // Set local references to mesh data
    const labelUList& owner = mesh_.owner();
    const labelUList& neighbour = mesh_.neighbour();
    const surfaceVectorField& Sf = mesh_.Sf();
    const surfaceVectorField& Cf = mesh_.Cf();
    const volVectorField& C = mesh_.C();

    // ... and reference to the internal field of the weighting factors
    scalarField& w = weights.primitiveFieldRef();

    static bool degenerateWarnedW = false;

    forAll(owner, facei)
    {
        // Defensive guard (see file-scope comment above): zero-area
        // face -> w = 0; original upstream code below would divide
        // by a zero distance sum.
        if (mag(Sf[facei]) < VSMALL)
        {
            w[facei] = 0;

            if (!degenerateWarnedW)
            {
                WarningInFunction
                    << "Degenerate face guard in makeWeights: zero-area"
                    << " face index " << facei
                    << "; weight set to 0 (face contributes nothing to"
                    << " surface integrals). Once per process." << endl;
                degenerateWarnedW = true;
            }

            continue;
        }

        // Note: mag in the dot-product.
        // For all valid meshes, the non-orthogonality will be less that
        // 90 deg and the dot-product will be positive.  For invalid
        // meshes (d & s <= 0), this will stabilise the calculation
        // but the result will be poor.
        const scalar SfdOwn = mag(Sf[facei]&(Cf[facei] - C[owner[facei]]));
        const scalar SfdNei = mag(Sf[facei]&(C[neighbour[facei]] - Cf[facei]));
        const scalar SfdOwnNei = SfdOwn + SfdNei;

        if (SfdNei/vGreat < SfdOwnNei)
        {
            w[facei] = SfdNei/SfdOwnNei;
        }
        else
        {
            const scalar dOwn = mag(Cf[facei] - C[owner[facei]]);
            const scalar dNei = mag(C[neighbour[facei]] - Cf[facei]);
            const scalar dOwnNei = dOwn + dNei;

            w[facei] = dNei/dOwnNei;
        }
    }

    surfaceScalarField::Boundary& wBf =
        weights.boundaryFieldRef();

    forAll(mesh_.boundary(), patchi)
    {
        mesh_.boundary()[patchi].makeWeights(wBf[patchi]);
    }

    if (debug)
    {
        Pout<< "surfaceInterpolation::makeWeights() : "
            << "Finished constructing weighting factors for face interpolation"
            << endl;
    }
}


void Foam::surfaceInterpolation::makeDeltaCoeffs() const
{
    if (debug)
    {
        Pout<< "surfaceInterpolation::makeDeltaCoeffs() : "
            << "Constructing interpolation factors array for face gradient"
            << endl;
    }

    // Force the construction of the weighting factors
    // needed to make sure deltaCoeffs are calculated for parallel runs.
    weights();

    deltaCoeffs_ = new surfaceScalarField
    (
        IOobject
        (
            "deltaCoeffs",
            mesh_.pointsInstance(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false // Do not register
        ),
        mesh_,
        dimless/dimLength
    );
    surfaceScalarField& deltaCoeffs = *deltaCoeffs_;


    // Set local references to mesh data
    const volVectorField& C = mesh_.C();
    const labelUList& owner = mesh_.owner();
    const labelUList& neighbour = mesh_.neighbour();
    const surfaceVectorField& Sf = mesh_.Sf();
    const surfaceScalarField& magSf = mesh_.magSf();

    static bool degenerateWarnedDC = false;

    forAll(owner, facei)
    {
        // Two-condition guard: face area ~ 0 OR cell-to-cell distance
        // ~ 0.  The second is the direct mechanical guard against the
        // 0/0 SIGFPE; the first is the semantic "is this a real face?"
        // check.  Either alone is insufficient — a stitcher artefact
        // can have area ~ 0 with non-zero cell distance, or vice versa.
        // Fallback is GREAT (= 1/SMALL) not 0: some BCs divide BY
        // deltaCoeff and need a finite divisor; GREAT mimics upstream
        // "1.0/tiny_distance = huge".  For flux integrals deltaCoeff
        // is multiplied by Sf which is zero on degenerate faces, so
        // physical contribution is zero regardless.
        if (mag(Sf[facei]) < VSMALL
         || mag(C[neighbour[facei]] - C[owner[facei]]) < VSMALL)
        {
            deltaCoeffs[facei] = GREAT;

            if (!degenerateWarnedDC)
            {
                WarningInFunction
                    << "Degenerate face guard in makeDeltaCoeffs"
                    << " (internal): face index " << facei
                    << "; deltaCoeff set to GREAT. Once per process."
                    << endl;
                degenerateWarnedDC = true;
            }

            continue;
        }
        deltaCoeffs[facei] = 1.0/mag(C[neighbour[facei]] - C[owner[facei]]);
    }

    surfaceScalarField::Boundary& deltaCoeffsBf =
        deltaCoeffs.boundaryFieldRef();

    forAll(deltaCoeffsBf, patchi)
    {
        // Per-face guarded path: the upstream vectorised expression
        //     deltaCoeffsBf[patchi] = 1.0/mag(mesh_.boundary()[patchi].delta());
        // raises SIGFPE if any face has zero owner-to-face distance,
        // which from-scratch AMI stitching can produce on degenerate
        // patch slices.  Expand into a per-face loop so degenerate
        // faces can take the GREAT fallback (see file-scope comment).
        // For valid meshes every face has finite distance and the
        // result is bit-identical to the vectorised upstream form.
        const vectorField patchDelta(mesh_.boundary()[patchi].delta());
        const scalarField& patchMagSf = magSf.boundaryField()[patchi];
        fvsPatchScalarField& bf = deltaCoeffsBf[patchi];

        forAll(bf, facei)
        {
            if (patchMagSf[facei] < VSMALL
             || mag(patchDelta[facei]) < VSMALL)
            {
                bf[facei] = GREAT;
                if (!degenerateWarnedDC)
                {
                    WarningInFunction
                        << "Degenerate face guard in makeDeltaCoeffs"
                        << " (boundary patch " << patchi
                        << ", face " << facei
                        << "): deltaCoeff set to GREAT."
                        << " Once per process." << endl;
                    degenerateWarnedDC = true;
                }
            }
            else
            {
                bf[facei] = 1.0/mag(patchDelta[facei]);
            }
        }

        // Original upstream expression — replaced by the per-face
        // guarded loop above:
        //     deltaCoeffsBf[patchi] = 1.0/mag(mesh_.boundary()[patchi].delta());
    }
}


void Foam::surfaceInterpolation::makeNonOrthDeltaCoeffs() const
{
    if (debug)
    {
        Pout<< "surfaceInterpolation::makeNonOrthDeltaCoeffs() : "
            << "Constructing interpolation factors array for face gradient"
            << endl;
    }

    // Force the construction of the weighting factors
    // needed to make sure deltaCoeffs are calculated for parallel runs.
    weights();

    nonOrthDeltaCoeffs_ = new surfaceScalarField
    (
        IOobject
        (
            "nonOrthDeltaCoeffs",
            mesh_.pointsInstance(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false // Do not register
        ),
        mesh_,
        dimless/dimLength
    );
    surfaceScalarField& nonOrthDeltaCoeffs = *nonOrthDeltaCoeffs_;


    // Set local references to mesh data
    const volVectorField& C = mesh_.C();
    const labelUList& owner = mesh_.owner();
    const labelUList& neighbour = mesh_.neighbour();
    const surfaceVectorField& Sf = mesh_.Sf();
    const surfaceScalarField& magSf = mesh_.magSf();

    static bool degenerateWarnedND = false;

    forAll(owner, facei)
    {
        // Two-condition guard (see makeDeltaCoeffs for rationale):
        // magSf ~ 0 (Sf/magSf below would be 0/0) OR cell-to-cell
        // distance ~ 0 (the 0.05*mag(delta) term in the max() below
        // would not save us if both arguments are 0).  Fallback is
        // GREAT, not 0, since downstream BCs may divide by this.
        if (magSf[facei] < VSMALL
         || mag(C[neighbour[facei]] - C[owner[facei]]) < VSMALL)
        {
            nonOrthDeltaCoeffs[facei] = GREAT;

            if (!degenerateWarnedND)
            {
                WarningInFunction
                    << "Degenerate face guard in makeNonOrthDeltaCoeffs"
                    << " (internal): face index " << facei
                    << "; nonOrthDeltaCoeff set to GREAT."
                    << " Once per process." << endl;
                degenerateWarnedND = true;
            }

            continue;
        }

        vector delta = C[neighbour[facei]] - C[owner[facei]];
        vector unitArea = Sf[facei]/magSf[facei];

        // Standard cell-centre distance form
        // NonOrthDeltaCoeffs[facei] = (unitArea & delta)/magSqr(delta);

        // Slightly under-relaxed form
        // NonOrthDeltaCoeffs[facei] = 1.0/mag(delta);

        // More under-relaxed form
        // NonOrthDeltaCoeffs[facei] = 1.0/(mag(unitArea & delta) + vSmall);

        // Stabilised form for bad meshes
        nonOrthDeltaCoeffs[facei] = 1.0/max(unitArea & delta, 0.05*mag(delta));
    }

    surfaceScalarField::Boundary& nonOrthDeltaCoeffsBf =
        nonOrthDeltaCoeffs.boundaryFieldRef();

    forAll(nonOrthDeltaCoeffsBf, patchi)
    {
        // Per-face guarded path: the upstream vectorised expression
        //     vectorField delta(mesh_.boundary()[patchi].delta());
        //     nonOrthDeltaCoeffsBf[patchi] =
        //         1.0/max(mesh_.boundary()[patchi].nf() & delta, 0.05*mag(delta));
        // raises SIGFPE when both arguments to max() are 0 on a
        // degenerate face.  Expand into a per-face loop so degenerate
        // faces take the GREAT fallback.
        const vectorField delta(mesh_.boundary()[patchi].delta());
        const vectorField nf(mesh_.boundary()[patchi].nf());
        const scalarField& patchMagSf = magSf.boundaryField()[patchi];
        fvsPatchScalarField& bf = nonOrthDeltaCoeffsBf[patchi];

        forAll(bf, facei)
        {
            if (patchMagSf[facei] < VSMALL
             || mag(delta[facei]) < VSMALL)
            {
                bf[facei] = GREAT;
                if (!degenerateWarnedND)
                {
                    WarningInFunction
                        << "Degenerate face guard in"
                        << " makeNonOrthDeltaCoeffs (boundary patch "
                        << patchi << ", face " << facei
                        << "): nonOrthDeltaCoeff set to GREAT."
                        << " Once per process." << endl;
                    degenerateWarnedND = true;
                }
            }
            else
            {
                bf[facei] =
                    1.0/max(nf[facei] & delta[facei], 0.05*mag(delta[facei]));
            }
        }

        // Original upstream expression — replaced by the per-face
        // guarded loop above:
        //     vectorField delta(mesh_.boundary()[patchi].delta());
        //     nonOrthDeltaCoeffsBf[patchi] =
        //         1.0/max(mesh_.boundary()[patchi].nf() & delta, 0.05*mag(delta));
    }
}


void Foam::surfaceInterpolation::makeNonOrthCorrectionVectors() const
{
    if (debug)
    {
        Pout<< "surfaceInterpolation::makeNonOrthCorrectionVectors() : "
            << "Constructing non-orthogonal correction vectors"
            << endl;
    }

    nonOrthCorrectionVectors_ = new surfaceVectorField
    (
        IOobject
        (
            "nonOrthCorrectionVectors",
            mesh_.pointsInstance(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false // Do not register
        ),
        mesh_,
        dimless
    );
    surfaceVectorField& corrVecs = *nonOrthCorrectionVectors_;

    // Set local references to mesh data
    const volVectorField& C = mesh_.C();
    const labelUList& owner = mesh_.owner();
    const labelUList& neighbour = mesh_.neighbour();
    const surfaceVectorField& Sf = mesh_.Sf();
    const surfaceScalarField& magSf = mesh_.magSf();
    const surfaceScalarField& NonOrthDeltaCoeffs = nonOrthDeltaCoeffs();

    static bool degenerateWarnedNC = false;

    forAll(owner, facei)
    {
        // Defensive guard against zero-area face: Sf/magSf below
        // would be 0/0.  Correction vector is Zero (no contribution).
        if (magSf[facei] < VSMALL)
        {
            corrVecs[facei] = Zero;

            if (!degenerateWarnedNC)
            {
                WarningInFunction
                    << "Degenerate face guard in"
                    << " makeNonOrthCorrectionVectors (internal):"
                    << " zero-area face index " << facei
                    << "; correction vector set to Zero."
                    << " Once per process." << endl;
                degenerateWarnedNC = true;
            }

            continue;
        }

        vector unitArea = Sf[facei]/magSf[facei];
        vector delta = C[neighbour[facei]] - C[owner[facei]];

        corrVecs[facei] = unitArea - delta*NonOrthDeltaCoeffs[facei];
    }

    // Boundary correction vectors set to zero for boundary patches
    // and calculated consistently with internal corrections for
    // coupled patches

    surfaceVectorField::Boundary& corrVecsBf =
        corrVecs.boundaryFieldRef();

    forAll(corrVecsBf, patchi)
    {
        fvsPatchVectorField& patchCorrVecs = corrVecsBf[patchi];

        if (!patchCorrVecs.coupled())
        {
            patchCorrVecs = Zero;
        }
        else
        {
            const fvsPatchScalarField& patchNonOrthDeltaCoeffs
                = NonOrthDeltaCoeffs.boundaryField()[patchi];

            const fvPatch& p = patchCorrVecs.patch();

            const vectorField patchDeltas(mesh_.boundary()[patchi].delta());

            forAll(p, patchFacei)
            {
                if (magSf.boundaryField()[patchi][patchFacei] < VSMALL)
                {
                    patchCorrVecs[patchFacei] = Zero;
                    if (!degenerateWarnedNC)
                    {
                        WarningInFunction
                            << "Degenerate face guard in"
                            << " makeNonOrthCorrectionVectors (coupled"
                            << " boundary patch " << patchi << ", face "
                            << patchFacei
                            << "): zero-area face; correction vector"
                            << " set to Zero. Once per process." << endl;
                        degenerateWarnedNC = true;
                    }
                    continue;
                }

                vector unitArea =
                    Sf.boundaryField()[patchi][patchFacei]
                   /magSf.boundaryField()[patchi][patchFacei];

                const vector& delta = patchDeltas[patchFacei];

                patchCorrVecs[patchFacei] =
                    unitArea - delta*patchNonOrthDeltaCoeffs[patchFacei];
            }
        }
    }

    if (debug)
    {
        Pout<< "surfaceInterpolation::makeNonOrthCorrectionVectors() : "
            << "Finished constructing non-orthogonal correction vectors"
            << endl;
    }
}


void Foam::surfaceInterpolation::printAllocated() const
{
    Pout<< "surfaceInterpolation allocated :" << endl;

    if (weights_)
    {
        Pout<< "    Weights" << endl;
    }

    if (deltaCoeffs_)
    {
        Pout<< "    Delta coefficients" << endl;
    }

    if (nonOrthDeltaCoeffs_)
    {
        Pout<< "    Non-orthogonal delta coefficients" << endl;
    }

    if (nonOrthCorrectionVectors_)
    {
        Pout<< "    Non-orthogonal correction vectors" << endl;
    }
}


// ************************************************************************* //
