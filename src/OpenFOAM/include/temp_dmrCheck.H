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

// Hybrid throttling.  DMR_AUTO is atomic with dmr_check(): once that
// call returns a non-NO_ACTION, DMR has already negotiated the new
// allocation with Slurm and the new process group is waiting, so we
// cannot defer DMR_AUTO.  Instead we throttle by deciding *whether to
// call dmr_check at all* on this step.
//
// Static-mesh cases: every reconfig boundary is safe.  Call
// DMR_AUTO(dmr_check) every step; DMR's own inhibitor controls cadence.
//
// Dynamic-mesh cases: from-scratch decompose can hit 0/0 in OF's AMI
// stitcher at moving-mesh positions where the patches have zero
// geometric overlap.  runTime.writeTime() boundaries are the case
// author's vetted points where the case state is fully consistent
// (mesh stitched, AMI weights well-defined, fields settled).  We:
//   - tick our own counter at every writeTime;
//   - call DMR_AUTO(dmr_check) only when the counter has reached
//     dmrDynInhibitor (read from DMR_DEFAULT_INHIBITOR at startup).
// DMR analytics for dynamic-mesh cases are therefore emitted at
// writeTime granularity (the only steps where dmr_check fires).
//
// dmrInit.H sets DMR's internal inhibitor to 0 for dynamic-mesh cases
// so each gated dmr_check actually evaluates the policy instead of
// being silently skipped by DMR's internal counter.
if (dmrEnabled
 && runTime.userTimeValue() < runTime.endTime().value())
{
    if (!dmrDynamicMesh)
    {
        DMR_AUTO
        (
            dmr_check(dmrPolicy),
            Foam::dmrCheckpoint(runTime, dmrAllRegions),
            (void)NULL,
            (void)NULL
        );
    }
    else
    {
        if (runTime.writeTime())
        {
            static int dmrDynCounter = 0;
            if (dmrDynCounter >= dmrDynInhibitor)
            {
                dmrDynCounter = 0;
                DMR_AUTO
                (
                    dmr_check(dmrPolicy),
                    Foam::dmrCheckpoint(runTime, dmrAllRegions),
                    (void)NULL,
                    (void)NULL
                );
            }
            else
            {
                ++dmrDynCounter;
            }
        }
    }
}
