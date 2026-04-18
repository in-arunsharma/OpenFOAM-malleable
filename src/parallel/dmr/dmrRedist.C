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

\*---------------------------------------------------------------------------*/

#include "dmrRedist.H"

#include "IOobjectList.H"
#include "HashSet.H"
#include "processorRunTimes.H"
#include "multiDomainDecomposition.H"
#include "domainDecomposition.H"
#include "fvFieldReconstructor.H"
#include "fvFieldDecomposer.H"
#include "polyMesh.H"
#include "Pstream.H"
#include "fileOperation.H"
#include "fieldTypes.H"

#include <mpi.h>
#include <sys/stat.h>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Module-level flag: set by dmrRestart(), consumed by dmrDecomposeInProcess().
// True only on the new process group, only after a DMR reconfiguration.
// Allows dmrDecomposeInProcess() to distinguish a restart from a fresh start.
static bool dmrRestartPending_ = false;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void dmrCheckpoint(Time& runTime, bool allRegions)
{
    // All ranks flush their fields to disk before rank 0 reconstructs.
    // This mirrors the write() that the solver already issued, but ensures
    // any pending output is committed even if the solver skipped the step.
    runTime.writeNow();

    // Barrier: wait for all parallel writes to land on disk.
    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        Info<< "DMR: Reconstructing to serial..." << nl;

        // Disable MPI mode immediately — any OF API call with parRun==true
        // inside a master-only block will issue MPI collectives that the
        // waiting ranks (at the barrier above) do not participate in, causing
        // deadlock.  With parRun==false, OF falls back to direct file I/O.
        // The shared Docker volume makes this safe: all nodes see the same
        // filesystem, and rank 0 is the only writer during reconstruction.
        const bool savedParRun = Pstream::parRun();
        Pstream::parRun() = false;

        // In a parallel run, runTime.path()   = rootPath/caseName
        //                                       = .../cavity/processor0  (local)
        //                 runTime.globalPath() = rootPath/globalCaseName
        //                                       = .../cavity             (global)
        // All reconstruction I/O targets the GLOBAL case root.
        const fileName globalCasePath = runTime.globalPath();
        const fileName globalCaseName = runTime.globalCaseName();

        // Count processor dirs with POSIX stat() — no OF API, no MPI.
        int nProcDirs = 0;
        {
            const std::string caseStr(globalCasePath.c_str());
            struct ::stat st;
            while
            (
                ::stat
                (
                    (caseStr + "/processor" + std::to_string(nProcDirs)).c_str(),
                    &st
                ) == 0
             && S_ISDIR(st.st_mode)
            )
            {
                ++nProcDirs;
            }
        }

        const wordList regionNames =
            allRegions
          ? runTime.regionNames()
          : wordList(1, polyMesh::defaultRegion);

        if (nProcDirs == 0)
        {
            WarningInFunction
                << "DMR: no processor directories found in '"
                << globalCasePath << "' — nothing to reconstruct." << nl;
        }
        else
        {
            // Build run-time objects using the proc count from decomposeParDict
            // (which still holds the pre-reconfiguration value).
            // Use rootPath + globalCaseName (not caseName which includes
            // "processorN" suffix in a parallel run).
            processorRunTimes procTimes
            (
                Time::controlDictName,
                runTime.rootPath(),
                globalCaseName,
                false,                                  // no function objects
                processorRunTimes::nProcsFrom::decomposeParDict
            );

            // Build the multi-region decomposition object.
            // meshPath = word::null → default (constant/polyMesh).
            multiDomainDecomposition regionMeshes
            (
                procTimes,
                word::null,
                regionNames
            );

            // Read processor meshes; reconstruct serial mesh if needed.
            regionMeshes.readReconstruct(false);

            // Reconstruct fields for every time present in processor dirs.
            const instantList times = procTimes.proc0Time().times();

            forAll(times, timei)
            {
                procTimes.setTime(times[timei], timei);

                Info<< "DMR: Reconstructing time "
                    << times[timei].name() << nl;

                // Update meshes for topology/motion changes.
                regionMeshes.readUpdateReconstruct();

                // Write the reconstructed (serial) mesh.
                regionMeshes.writeComplete(false);

                // Reconstruct FV fields region by region.
                forAll(regionNames, regioni)
                {
                    const RegionRef<domainDecomposition> meshes =
                        regionMeshes[regioni];

                    IOobjectList objects
                    (
                        meshes().procMeshes()[0],
                        procTimes.proc0Time().name()
                    );

                    if
                    (
                        fvFieldReconstructor::reconstructs
                        (
                            objects,
                            wordHashSet()
                        )
                    )
                    {
                        fvFieldReconstructor fvReconstructor
                        (
                            meshes().completeMesh(),
                            meshes().procMeshes(),
                            meshes().procFaceAddressing(),
                            meshes().procCellAddressing(),
                            meshes().procFaceAddressingBf()
                        );

                        #define DO_FV_INTERNAL(Type, nullArg)               \
                            fvReconstructor                                  \
                               .reconstructVolInternalFields<Type>           \
                                (objects, wordHashSet());
                        FOR_ALL_FIELD_TYPES(DO_FV_INTERNAL)
                        #undef DO_FV_INTERNAL

                        #define DO_FV_VOL(Type, nullArg)                    \
                            fvReconstructor                                  \
                               .reconstructVolFields<Type>                   \
                                (objects, wordHashSet());
                        FOR_ALL_FIELD_TYPES(DO_FV_VOL)
                        #undef DO_FV_VOL

                        #define DO_FV_SURFACE(Type, nullArg)                \
                            fvReconstructor                                  \
                               .reconstructFvSurfaceFields<Type>             \
                                (objects, wordHashSet());
                        FOR_ALL_FIELD_TYPES(DO_FV_SURFACE)
                        #undef DO_FV_SURFACE
                    }
                }
            }

            // Remove all processor* directories now that the serial case
            // is complete.  The DMR new-process set will re-decompose.
            // Use globalCasePath — runTime.path() is the local processor path.

            const fileNameList dirs
            (
                fileHandler().readDir(globalCasePath, fileType::directory)
            );

            forAll(dirs, diri)
            {
                const fileName& d = dirs[diri];

                // Match "processor<integer>" directories only.
                if (d.find("processor") == 0)
                {
                    fileName num(d.substr(9));
                    label proci = -1;
                    if (Foam::read(num.c_str(), proci))
                    {
                        fileHandler().rmDir(globalCasePath/d);
                    }
                }
            }

            Info<< "DMR: Reconstruction complete." << nl;
        }

        // Restore MPI mode — covers both the warning and reconstruction paths.
        Pstream::parRun() = savedParRun;
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void dmrRestart(const std::string& casePath, bool allRegions)
{
    // Phase 1 of the two-phase restart — called inside DMR_AUTO, while
    // DMR_INTERCOMM is still live.  Only perform fast, MPI-free work here:
    // update the two config files so the new decomposition uses the correct
    // process count, then set the pending flag so dmrDecomposeInProcess()
    // (Phase 2, called AFTER DMR_AUTO) knows it must decompose.
    //
    // No MPI_Barrier here — dmr_reconfigure() (called by DMR_AUTO immediately
    // after this function returns) handles the inter-group synchronisation.
    // The real barrier is inside dmrDecomposeInProcess().

    int newSize, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &newSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    if (myRank == 0)
    {
        std::fprintf
        (
            stderr,
            "DMR: Restarting with %d processes (dicts update, decompose deferred).\n",
            newSize
        );

        // ---- Update decomposeParDict ----------------------------------------

        const std::string dictPath = casePath + "/system/decomposeParDict";

        if
        (
           !dmrReplaceLine
            (
                dictPath,
                "numberOfSubdomains",
                "numberOfSubdomains  " + std::to_string(newSize) + ";"
            )
        )
        {
            std::fprintf
            (
                stderr,
                "DMR: FATAL — failed to update numberOfSubdomains\n"
            );
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Force scotch: geometric decomposers (hierarchical, simple) require
        // n-coefficients that break when DMR changes the process count.
        if (!dmrReplaceLine(dictPath, "decomposer", "decomposer      scotch;"))
        {
            // Older-style dicts may use 'method' instead of 'decomposer'
            dmrReplaceLine(dictPath, "method", "method          scotch;");
        }

        // ---- Update controlDict ---------------------------------------------

        const std::string ctrlDictPath = casePath + "/system/controlDict";

        if
        (
           !dmrReplaceLine
            (
                ctrlDictPath,
                "startFrom",
                "startFrom       latestTime;"
            )
        )
        {
            std::fprintf
            (
                stderr,
                "DMR: WARNING — could not set startFrom in controlDict\n"
            );
        }
    }

    // All ranks set the flag so dmrDecomposeInProcess() knows this is a
    // restart (not a fresh start where it would be a no-op).
    dmrRestartPending_ = true;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void dmrDecomposeInProcess(const std::string& casePath, bool allRegions)
{
    // Phase 2 of the two-phase restart — called AFTER DMR_AUTO returns.
    // By this point dmr_reconfigure() has already freed DMR_INTERCOMM and
    // MPI_COMM_WORLD is in a clean state with only the new process group.
    //
    // For fresh starts (no prior reconfiguration) dmrRestartPending_ is false
    // and this function is a no-op.

    if (!dmrRestartPending_) return;
    dmrRestartPending_ = false;

    int newSize, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &newSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    if (myRank == 0)
    {
        const std::string restartTime = dmrFindLatestTime(casePath);

        if (allRegions)
        {
            // Multi-region: in-process region enumeration requires a solver
            // runTime object, which is not available here.  Fall back to the
            // subprocess approach so foamMultiRun cases are not broken.
            std::fprintf
            (
                stderr,
                "DMR: Multi-region — subprocess decompose for %d procs (time=%s).\n",
                newSize,
                restartTime.c_str()
            );

            const std::string decomposeCmd =
                "decomposePar"
                " -force"
                " -time " + restartTime +
                " -cellProc"
                " -allRegions"
                " -case " + casePath
              + " > /tmp/dmr_decompose.log 2>&1";

            const int ret = std::system(decomposeCmd.c_str());

            if (ret != 0)
            {
                std::fprintf
                (
                    stderr,
                    "DMR: FATAL — decomposePar failed (exit %d);"
                    " see /tmp/dmr_decompose.log\n",
                    ret
                );
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else
        {
            // Single-region: in-process decompose using the OpenFOAM API.
            //
            // DMR_INTERCOMM has been freed before this function is called.
            // With parRun=false, UPstream::allocateCommunicator does NOT call
            // MPI_Comm_create and freeCommunicator does NOT call MPI_Comm_free
            // — the OpenMPI context ID pool is completely unaffected.
            // All I/O goes directly to the shared filesystem.
            std::fprintf
            (
                stderr,
                "DMR: In-process decompose for %d procs (time=%s)...\n",
                newSize,
                restartTime.c_str()
            );

            const bool savedParRun = Pstream::parRun();
            Pstream::parRun() = false;

            {
                const fileName foamCase(casePath);
                const wordList regionNames(1, polyMesh::defaultRegion);

                processorRunTimes procTimes
                (
                    Time::controlDictName,
                    foamCase.path(),
                    foamCase.name(),
                    false,
                    processorRunTimes::nProcsFrom::decomposeParDict
                );

                multiDomainDecomposition regionMeshes
                (
                    procTimes,
                    word::null,
                    regionNames
                );

                // Read the complete (serial) mesh and build the decomposition
                // map in memory.
                regionMeshes.readDecompose(false);

                // Advance to the restart time so decomposed fields land in
                // the correct time directory.
                const instantList completeTimes =
                    procTimes.completeTime().times();
                instant restartInst("0");
                forAll(completeTimes, timei)
                {
                    if (completeTimes[timei].name() == word(restartTime))
                    {
                        restartInst = completeTimes[timei];
                        break;
                    }
                }
                procTimes.setTime(restartInst, 0);

                // Update decomposition for the restart time (handles mesh
                // motion — writes moved points per processor), then write all
                // processor directories to disk.
                regionMeshes.readUpdateDecompose();
                regionMeshes.writeProcs(false);
            }

            Pstream::parRun() = savedParRun;
            std::fprintf(stderr, "DMR: In-process decompose complete.\n");
        }
    }

    // Barrier: all new-group ranks synchronise before proceeding to
    // setRootCase.H, which re-initialises OpenFOAM in parallel mode.
    MPI_Barrier(MPI_COMM_WORLD);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
