// DMR reconfiguration check — include after runTime.write().
// Requires FOAM_DMR_ENABLED=1 and DMR_CHECK_INTERVAL to be set.

if (dmrEnabled)
{
    static int dmrStepCounter = 0;
    static int dmrCheckInterval = -1;

    if (dmrCheckInterval < 0)
    {
        const char* env = std::getenv("DMR_CHECK_INTERVAL");
        if (!env || std::atoi(env) <= 0)
        {
            FatalErrorInFunction
                << "DMR is enabled but DMR_CHECK_INTERVAL is not set.\n"
                << "Export DMR_CHECK_INTERVAL=<N> (number of time steps "
                << "between reconfiguration checks)."
                << exit(FatalError);
        }
        dmrCheckInterval = std::atoi(env);
        Info<< "DMR: check interval=" << dmrCheckInterval
            << " steps" << nl;
    }

    if (++dmrStepCounter >= dmrCheckInterval)
    {
        dmrStepCounter = 0;

        // Skip if the simulation is about to end — avoid a pointless
        // reconstruct+decompose on the final step.
        if (runTime.timeOutputValue() >= runTime.endTime().value())
        {
            Info<< "DMR: At endTime, skipping reconfiguration check."
                << nl;
        }
        else
        {
            Info<< "DMR: Checking for reconfiguration..." << nl;

            DMR_AUTO(
                dmr_check(ROUND_POLICY),
                dmrCheckpoint(runTime, dmrAllRegions),
                (void)NULL,
                (void)NULL
            );
        }
    }
}
