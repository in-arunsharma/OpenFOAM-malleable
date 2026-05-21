// DMR reconfiguration check — include after runTime.write().
// dmr_check() is called every time step; cadence is owned by the DMR
// inhibitor (DMR_DEFAULT_INHIBITOR), not by an OpenFOAM-side counter.
// When the policy fires, dmrCheckpoint() forces a runTime.writeNow().

if (dmrEnabled)
{
    DMR_AUTO
    (
        dmr_check(dmrPolicy),
        Foam::dmrCheckpoint(runTime, dmrAllRegions),
        (void)NULL,
        (void)NULL
    );
}
