// DMR reconfiguration check — include after runTime.write().
//
// Cadence is owned entirely by DMR's inhibitor mechanism
// (DMR_DEFAULT_INHIBITOR env var, or compile default, or the runtime
// API dmr_set_reconf_step_inhibitor).  Inhibited calls return
// immediately with a counter compare; the policy and any
// reconfiguration only fire on the (N+1)th call when the inhibitor
// is N.  This matches the BSC integration pattern used by other
// malleable applications (e.g. Alya) and keeps all cadence state in
// the DMR namespace.
//
// What differs between static and dynamic mesh is *what counts as
// one opportunity*, not who does the counting:
//
//   Static mesh   — every time step is a safe reconfiguration
//                   boundary, so dmr_check is called every step and
//                   DMR's inhibitor counts steps.
//
//   Dynamic mesh  — a from-scratch decompose can hit 0/0 in OF's AMI
//                   stitcher at moving-mesh positions where patches
//                   have zero geometric overlap.  runTime.writeTime()
//                   is the case author's vetted "fully consistent
//                   state" boundary (mesh stitched, AMI weights
//                   well-defined, fields settled), so dmr_check is
//                   called only at writeTime; DMR's inhibitor then
//                   counts writeTime events.
//
// DMR_AUTO is atomic with dmr_check(): once that call returns a
// non-NO_ACTION, DMR has already negotiated the new allocation with
// Slurm and the new process group is waiting, so we cannot defer
// DMR_AUTO.  We throttle by deciding *whether to call dmr_check at
// all* on this step.

const bool dmrCheckThisStep =
    !dmrDynamicMesh || runTime.writeTime();

if (dmrEnabled
 && dmrCheckThisStep
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
