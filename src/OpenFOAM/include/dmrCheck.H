// DMR reconfiguration check — include after runTime.write().
//
// dmr_check() is called every time step.  Cadence is owned entirely
// by DMR's inhibitor mechanism (DMR_DEFAULT_INHIBITOR env var, or
// compile default, or the runtime API
// dmr_set_reconf_step_inhibitor): inhibited calls return immediately
// with a counter compare; the policy and any reconfiguration only
// fire on the (N+1)th call when the inhibitor is N.  This matches
// the BSC integration pattern used by other malleable applications
// (e.g. Alya) — single source of truth for cadence in the DMR
// namespace, no OpenFOAM-side counters.
//
// Same handling for static and dynamic mesh.  The earlier dynamic-
// mesh-specific gating to runTime.writeTime() was defensive: the
// assumption was that writeIntervals are case-author-vetted "safe"
// boundaries.  That assumption was incorrect — writeIntervals are
// an output convention, with no relation to AMI stitchability.  The
// gate also produced unreliable cadence after a restart, because
// adjustableRunTime's writeTimeIndex tracking does not always fire
// writeTime() at every writeInterval in the new process group.
//
// When the policy fires, dmrCheckpoint() (the OLD-group branch of
// DMR_AUTO) forces a runTime.writeNow() at the current step, so the
// write does not depend on runTime.writeTime() being true.
if (dmrEnabled
 && runTime.userTimeValue() < runTime.endTime().value())
{
    DMR_AUTO
    (
        dmr_check(dmrPolicy),
        Foam::dmrCheckpoint(runTime, dmrAllRegions),
        (void)NULL,
        (void)NULL
    );
}
