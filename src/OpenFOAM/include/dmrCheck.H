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
//
// Note: we do NOT additionally guard on `runTime.userTimeValue() <
// runTime.endTime().value()` here.  That comparison silently mixes
// units when the case uses a non-trivial userTime (e.g. `userTime
// engine` converting CAD ↔ seconds): userTimeValue() returns the
// numeric value in user units (CAD), while endTime().value() returns
// the numeric value in raw seconds.  At the moment the user-time
// scalar overtakes the raw-time scalar (CAD=4.0 vs raw 4.0 s for a
// typical engine RPM, for example), the comparison becomes false and
// dmr_check stops firing for the rest of the run.  The outer
// `while (pimple.run(runTime))` already correctly gates entry to this
// code by end-of-run, so no additional guard is needed.
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
