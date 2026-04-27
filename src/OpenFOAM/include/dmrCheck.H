// DMR reconfiguration check — include after runTime.write().
//
// dmr_check() is called every time step.  DMR's own inhibitor mechanism
// (DMR_DEFAULT_INHIBITOR env var, or compile default, or the runtime
// API dmr_set_reconf_step_inhibitor) decides how often the call actually
// evaluates the policy.  Inhibited calls return immediately with a
// counter compare; the policy and any reconfiguration only fire on the
// (N+1)th call when the inhibitor is N.  This matches the BSC integration
// pattern used by other malleable applications (e.g. Alya) and keeps the
// cadence configuration in the DMR namespace, so OpenFOAM does not
// duplicate state with the DMR library.

if (dmrEnabled)
{
    // Skip if the simulation has already reached endTime — a reconfig
    // here would reconstruct + decompose with no further time stepping
    // to amortise the cost.
    if (runTime.userTimeValue() < runTime.endTime().value())
    {
        DMR_AUTO
        (
            dmr_check(dmrPolicy),
            Foam::dmrCheckpoint(runTime, dmrAllRegions),
            (void)NULL,
            (void)NULL
        );
    }
}
