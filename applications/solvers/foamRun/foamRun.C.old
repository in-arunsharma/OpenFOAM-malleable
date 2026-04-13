/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2025 OpenFOAM Foundation
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
    foamRun

Description
    Loads and executes an OpenFOAM solver module either specified by the
    optional \c solver entry in the \c controlDict or as a command-line
    argument.

    Uses the flexible PIMPLE (PISO-SIMPLE) solution for time-resolved and
    pseudo-transient and steady simulations.

Usage
    \b foamRun [OPTION]

      - \par -solver <name>
        Solver name

      - \par -libs '(\"lib1.so\" ... \"libN.so\")'
        Specify the additional libraries loaded

    Example usage:
      - To run a \c rhoPimpleFoam case by specifying the solver on the
        command line:
        \verbatim
            foamRun -solver fluid
        \endverbatim

      - To update and run a \c rhoPimpleFoam case add the following entry to
        the controlDict:
        \verbatim
            solver          fluid;
        \endverbatim
        then execute \c foamRun

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "solver.H"
#include "pimpleSingleRegionControl.H"
#include "setDeltaT.H"

#ifdef FOAM_USE_DMR
    #include "foamDmr.H"
    #include "foamDmrRedist.H"
#endif

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addOption
    (
        "solver",
        "name",
        "Solver name"
    );

    #ifdef FOAM_USE_DMR
    // --- DMR: Initialize dynamic resource management ---
    // Must run BEFORE setRootCase.H because argList validates that the
    // processor count matches decomposeParDict. On DMR restart the count
    // has changed, so dmrRestart() must update the dict first.
    //
    // DMR requires MPI to be already initialized, so we call MPI_Init here.
    // OpenFOAM's UPstream::init (inside setRootCase.H) will detect that
    // MPI is already initialized and skip the duplicate MPI_Init.
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    }

    // Parse -case from argv (OpenFOAM's argList is not constructed yet).
    std::string dmrCasePath = ".";
    for (int i = 1; i < argc - 1; ++i)
    {
        if (std::string(argv[i]) == "-case")
        {
            dmrCasePath = argv[i + 1];
            break;
        }
    }

    DMR_AUTO(
        dmr_init(argc, argv),
        (void)NULL,                    // checkpoint: not needed at init
        dmrRestart(dmrCasePath),       // restart: decompose for new proc count
        (void)NULL                     // finalize: not needed at init
    );

    // Configure DMR policy from environment variables.
    // The Docker DMR library (v2.0.0) does not read these env vars
    // natively, so we bridge them to the C API here.
    {
        const char* envMax = std::getenv("DMR_DEFAULT_POLICY_MAX");
        if (envMax) dmr_set_policy_max_nodes(std::atoi(envMax));

        const char* envMin = std::getenv("DMR_DEFAULT_POLICY_MIN");
        if (envMin) dmr_set_policy_min_nodes(std::atoi(envMin));
    }
    #endif

    #include "setRootCase.H"

    #include "createTime.H"

    // Read the solverName from the optional solver entry in controlDict
    word solverName
    (
        runTime.controlDict().lookupOrDefault("solver", word::null)
    );

    // Optionally reset the solver name from the -solver command-line argument
    args.optionReadIfPresent("solver", solverName);

    // Check the solverName has been set
    if (solverName == word::null)
    {
        args.printUsage();

        FatalErrorIn(args.executable())
            << "solver not specified in the controlDict or on the command-line"
            << exit(FatalError);
    }
    else
    {
        // Load the solver library
        solver::load(solverName);
    }

    // Create the default single region mesh
    #include "createMesh.H"

    // Instantiate the selected solver
    autoPtr<solver> solverPtr(solver::New(solverName, mesh));
    solver& solver = solverPtr();

    // Create the outer PIMPLE loop and control structure
    pimpleSingleRegionControl pimple(solver.pimple);

    // Set the initial time-step
    setDeltaT(runTime, solver);

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    #ifdef FOAM_USE_DMR
    // --- DMR: Policy (min/max nodes, inhibitor, PPN) is configured via
    //     environment variables and read by the DMR library directly.
    //     Only the check interval is application-specific. ---
    const label dmrCheckInterval = []()
    {
        const char* env = std::getenv("DMR_CHECK_INTERVAL");
        return env ? std::atoi(env) : 100;
    }();

    Info<< "DMR: check interval=" << dmrCheckInterval << " steps" << nl;

    label dmrStepCounter = 0;
    #endif

    Info<< nl << "Starting time loop\n" << endl;

    while (pimple.run(runTime))
    {
        solver.preSolve();

        // Adjust the time-step according to the solver maxDeltaT
        adjustDeltaT(runTime, solver);

        runTime++;

        Info<< "Time = " << runTime.userTimeName() << nl << endl;

        // PIMPLE corrector loop
        while (pimple.loop())
        {
            if (solver.pimple.flow())
            {
                solver.moveMesh();
                solver.motionCorrector();
            }

            if (solver.pimple.models())
            {
                solver.fvModels().correct();
            }

            solver.prePredictor();

            if (solver.pimple.predictTransport())
            {
                if (solver.pimple.flow())
                {
                    solver.momentumTransportPredictor();
                }

                if (solver.pimple.thermophysics())
                {
                    solver.thermophysicalTransportPredictor();
                }
            }

            if (solver.pimple.flow())
            {
                solver.momentumPredictor();
            }

            if (solver.pimple.thermophysics())
            {
                solver.thermophysicalPredictor();
            }

            if (solver.pimple.flow())
            {
                solver.pressureCorrector();
            }

            if (solver.pimple.correctTransport())
            {
                if (solver.pimple.flow())
                {
                    solver.momentumTransportCorrector();
                }

                if (solver.pimple.thermophysics())
                {
                    solver.thermophysicalTransportCorrector();
                }
            }
        }

        solver.postSolve();

        runTime.write();

        #ifdef FOAM_USE_DMR
        // --- DMR: Check for reconfiguration every dmrCheckInterval steps ---
        dmrStepCounter++;
        if (dmrStepCounter >= dmrCheckInterval)
        {
            dmrStepCounter = 0;

            Info<< "DMR: Checking for reconfiguration..." << nl;

            // ROUND_POLICY = debug policy, cycles node count up/down.
            // For production, switch to CE_POLICY (TALP communication
            // efficiency). If DMR decides to reconfigure:
            //   checkpoint: runTime.writeNow() writes all fields
            //   then the application relaunches with new process count.
            //   OpenFOAM restarts from latestTime automatically.
            DMR_AUTO(
                dmr_check(ROUND_POLICY),
                dmrCheckpoint(runTime),     // checkpoint: write + reconstruct
                (void)NULL,                 // restart: not used here
                (void)NULL                  // finalize: not used here
            );
        }
        #endif

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    #ifdef FOAM_USE_DMR
    // --- DMR: Clean shutdown ---
    DMR_AUTO(dmr_finalize(), (void)NULL, (void)NULL, (void)NULL);
    #endif

    return 0;
}


// ************************************************************************* //
