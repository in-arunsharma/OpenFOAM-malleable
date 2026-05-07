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

#include <cstdlib>
#include <string>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(surfaceInterpolation, 0);
}


// * * * * * * * * * * * * Local helpers (DMR guard) * * * * * * * * * * * * //

namespace
{

// Defensive guard against degenerate faces produced by from-scratch AMI
// stitching during DMR-driven reconfiguration (libfoamDmr).
//
// Activation: dormant unless the environment variable
//   FOAM_DMR_AMI_DEGENERATE_GUARD=active
// is set in the calling process.  libfoamDmr sets this variable as a
// command-line prefix on the decomposePar subprocess it spawns during a
// reconfiguration; the variable lives only inside that one subprocess and
// is never visible to foamRun, foamMultiRun, or any other OpenFOAM tool.
// In every non-DMR context the guard is dormant and OpenFOAM's behaviour
// is bit-identical to upstream.
//
// What it guards: fvMeshStitcher::connectThis can construct synthetic
// faces with zero area whose owner cell centre, neighbour cell centre,
// and face centre all coincide, when the geometric overlap between two
// non-conformal AMI patches has collapsed to zero (e.g. fully-closed
// engine valves).  Several geometric coefficient computations in
// surfaceInterpolation then divide by zero (1.0/distance, 1.0/area,
// Sf/|Sf|), raising SIGFPE inside decomposePar.  Such a face contributes
// nothing to any flux or gradient because its area appears multiplicatively
// in every surface integral, so the coefficient values are mathematically
// irrelevant; setting them to zero is the unique value consistent with
// "no contribution from this face".
bool dmrAmiDegenerateGuardActive()
{
    static const bool active = []()
    {
        const char* env = std::getenv("FOAM_DMR_AMI_DEGENERATE_GUARD");
        return env != nullptr && std::string(env) == "active";
    }();
    return active;
}

} // End anonymous namespace


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

    // DMR degenerate-face guard (see anonymous-namespace helper above).
    const bool dmrGuard = dmrAmiDegenerateGuardActive();
    static bool dmrGuardWarnedW = false;

    forAll(owner, facei)
    {
        if (dmrGuard && mag(Sf[facei]) < VSMALL)
        {
            w[facei] = 0;

            if (!dmrGuardWarnedW)
            {
                WarningInFunction
                    << "FOAM_DMR_AMI_DEGENERATE_GUARD active in"
                    << " makeWeights: zero-area face index " << facei
                    << "; weight set to 0 (face contributes nothing to"
                    << " surface integrals). Once per process." << endl;
                dmrGuardWarnedW = true;
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

    // DMR degenerate-face guard (see anonymous-namespace helper above).
    const bool dmrGuard = dmrAmiDegenerateGuardActive();
    static bool dmrGuardWarnedDC = false;

    forAll(owner, facei)
    {
        // Two-condition guard: face is geometrically degenerate
        // (mag(Sf) ~ 0) OR the divisor about to be used is degenerate
        // (cell-to-cell distance ~ 0).  The second condition is the
        // direct mechanical guard against the 0/0 SIGFPE; the first
        // is the semantic "is this a real face?" check.  Either
        // alone is insufficient — a stitcher artefact can have area
        // ~0 with non-zero cell distance, or non-zero area with
        // coincident cell centres.  Both must be covered.
        if (dmrGuard
         && (mag(Sf[facei]) < VSMALL
          || mag(C[neighbour[facei]] - C[owner[facei]]) < VSMALL))
        {
            // Set deltaCoeff to GREAT (= 1/SMALL ~ 4.5e15) rather than
            // 0.  GREAT mimics the original "1.0/tiny_distance = huge"
            // behaviour upstream OpenFOAM produces on near-degenerate
            // geometry: downstream code that does 1/deltaCoeff or
            // gradient/deltaCoeff (e.g. directionMixedFvPatchField)
            // gets a numerically tiny but finite result, no SIGFPE.
            // For internal flux integrals the deltaCoeff is multiplied
            // by Sf (zero on degenerate faces), so the contribution is
            // still zero regardless.
            deltaCoeffs[facei] = GREAT;

            if (!dmrGuardWarnedDC)
            {
                WarningInFunction
                    << "FOAM_DMR_AMI_DEGENERATE_GUARD active in"
                    << " makeDeltaCoeffs (internal): degenerate face"
                    << " index " << facei
                    << "; deltaCoeff set to GREAT. Once per process."
                    << endl;
                dmrGuardWarnedDC = true;
            }

            continue;
        }
        deltaCoeffs[facei] = 1.0/mag(C[neighbour[facei]] - C[owner[facei]]);
    }

    surfaceScalarField::Boundary& deltaCoeffsBf =
        deltaCoeffs.boundaryFieldRef();

    forAll(deltaCoeffsBf, patchi)
    {
        if (dmrGuard)
        {
            // Per-face guarded path: the upstream vectorised expression
            // 1.0/mag(boundary().delta()) raises SIGFPE if any face has
            // zero owner-to-face distance.  Expand into a per-face loop
            // so we can skip degenerate faces.
            const vectorField patchDelta(mesh_.boundary()[patchi].delta());
            const scalarField& patchMagSf = magSf.boundaryField()[patchi];
            fvsPatchScalarField& bf = deltaCoeffsBf[patchi];

            forAll(bf, facei)
            {
                // Two-condition guard (see makeDeltaCoeffs internal
                // loop above for the same rationale): face area ~ 0
                // OR cell-to-face distance ~ 0.  Either alone is
                // insufficient.
                if (patchMagSf[facei] < VSMALL
                 || mag(patchDelta[facei]) < VSMALL)
                {
                    // Set to GREAT (~1/SMALL) rather than 0; some BCs
                    // (e.g. directionMixedFvPatchField) divide BY
                    // deltaCoeff and would SIGFPE on 0.  GREAT mimics
                    // the upstream "1/tiny_distance = huge" behaviour.
                    bf[facei] = GREAT;
                    if (!dmrGuardWarnedDC)
                    {
                        WarningInFunction
                            << "FOAM_DMR_AMI_DEGENERATE_GUARD active in"
                            << " makeDeltaCoeffs (boundary patch "
                            << patchi << ", face " << facei
                            << "): degenerate face; deltaCoeff set to"
                            << " GREAT. Once per process." << endl;
                        dmrGuardWarnedDC = true;
                    }
                }
                else
                {
                    bf[facei] = 1.0/mag(patchDelta[facei]);
                }
            }
        }
        else
        {
            // Upstream path, unchanged.
            deltaCoeffsBf[patchi] = 1.0/mag(mesh_.boundary()[patchi].delta());
        }
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

    // DMR degenerate-face guard (see anonymous-namespace helper above).
    const bool dmrGuard = dmrAmiDegenerateGuardActive();
    static bool dmrGuardWarnedND = false;

    forAll(owner, facei)
    {
        // Two-condition guard (see makeDeltaCoeffs for rationale):
        // magSf ~ 0 (Sf/magSf below would be 0/0) OR cell-to-cell
        // distance ~ 0 (the 0.05*mag(delta) term in the max() below
        // would not save us if both arguments are 0).
        if (dmrGuard
         && (magSf[facei] < VSMALL
          || mag(C[neighbour[facei]] - C[owner[facei]]) < VSMALL))
        {
            // Set to GREAT not 0; downstream BCs and corrections may
            // divide by nonOrthDeltaCoeff.  See the same rationale in
            // makeDeltaCoeffs above.
            nonOrthDeltaCoeffs[facei] = GREAT;

            if (!dmrGuardWarnedND)
            {
                WarningInFunction
                    << "FOAM_DMR_AMI_DEGENERATE_GUARD active in"
                    << " makeNonOrthDeltaCoeffs (internal): degenerate"
                    << " face index " << facei
                    << "; nonOrthDeltaCoeff set to GREAT."
                    << " Once per process." << endl;
                dmrGuardWarnedND = true;
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
        if (dmrGuard)
        {
            // Per-face guarded path: skip boundary faces with zero area
            // before the 1.0/max(...) division can blow up.
            const vectorField delta(mesh_.boundary()[patchi].delta());
            const vectorField nf(mesh_.boundary()[patchi].nf());
            const scalarField& patchMagSf = magSf.boundaryField()[patchi];
            fvsPatchScalarField& bf = nonOrthDeltaCoeffsBf[patchi];

            forAll(bf, facei)
            {
                // Two-condition guard: magSf ~ 0 OR mag(delta) ~ 0.
                // The 1.0/max(... , 0.05*mag(delta)) form is safe
                // only while mag(delta) is well above zero; when both
                // arguments to max() are 0 the division crashes.
                if (patchMagSf[facei] < VSMALL
                 || mag(delta[facei]) < VSMALL)
                {
                    // GREAT, not 0: downstream BC code divides by
                    // nonOrthDeltaCoeff.  Same rationale as
                    // makeDeltaCoeffs (boundary).
                    bf[facei] = GREAT;
                    if (!dmrGuardWarnedND)
                    {
                        WarningInFunction
                            << "FOAM_DMR_AMI_DEGENERATE_GUARD active in"
                            << " makeNonOrthDeltaCoeffs (boundary patch "
                            << patchi << ", face " << facei
                            << "): degenerate face; nonOrthDeltaCoeff"
                            << " set to GREAT. Once per process." << endl;
                        dmrGuardWarnedND = true;
                    }
                }
                else
                {
                    bf[facei] =
                        1.0/max(nf[facei] & delta[facei], 0.05*mag(delta[facei]));
                }
            }
        }
        else
        {
            // Upstream path, unchanged.
            vectorField delta(mesh_.boundary()[patchi].delta());

            nonOrthDeltaCoeffsBf[patchi] =
                1.0/max(mesh_.boundary()[patchi].nf() & delta, 0.05*mag(delta));
        }
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

    // DMR degenerate-face guard (see anonymous-namespace helper above).
    const bool dmrGuard = dmrAmiDegenerateGuardActive();
    static bool dmrGuardWarnedNC = false;

    forAll(owner, facei)
    {
        if (dmrGuard && magSf[facei] < VSMALL)
        {
            corrVecs[facei] = Zero;

            if (!dmrGuardWarnedNC)
            {
                WarningInFunction
                    << "FOAM_DMR_AMI_DEGENERATE_GUARD active in"
                    << " makeNonOrthCorrectionVectors (internal):"
                    << " zero-area face index " << facei
                    << "; correction vector set to Zero."
                    << " Once per process." << endl;
                dmrGuardWarnedNC = true;
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
                if (dmrGuard
                 && magSf.boundaryField()[patchi][patchFacei] < VSMALL)
                {
                    patchCorrVecs[patchFacei] = Zero;
                    if (!dmrGuardWarnedNC)
                    {
                        WarningInFunction
                            << "FOAM_DMR_AMI_DEGENERATE_GUARD active in"
                            << " makeNonOrthCorrectionVectors (coupled"
                            << " boundary patch " << patchi << ", face "
                            << patchFacei
                            << "): zero-area face; correction vector"
                            << " set to Zero. Once per process." << endl;
                        dmrGuardWarnedNC = true;
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
