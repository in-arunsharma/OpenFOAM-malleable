// DMR clean shutdown — include after the time loop.
// Reconstructs any remaining processor dirs so the final case is
// fully serial and consistent with earlier checkpoint reconstructions.

if (dmrEnabled)
{
    DMR_AUTO(dmr_finalize(), (void)NULL, (void)NULL, (void)NULL);

    // Final reconstruction: the time loop already wrote fields via
    // runTime.write(), so we just need to merge processor dirs.
    MPI_Barrier(MPI_COMM_WORLD);

    if (Pstream::master())
    {
        const fileName casePath = runTime.globalPath();

        Info<< "DMR: Final reconstruction to serial..." << nl;

        std::string cmd("reconstructPar -case ");
        cmd += casePath;
        cmd += " -newTimes -rm";
        if (dmrAllRegions) cmd += " -allRegions";

        const int ret = ::system(cmd.c_str());

        if (ret != 0)
        {
            WarningInFunction
                << "Final reconstructPar failed (exit code " << ret
                << "). Processor directories may remain." << nl;
        }
        else
        {
            Info<< "DMR: Final reconstruction complete." << nl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}
