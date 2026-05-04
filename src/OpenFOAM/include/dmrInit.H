// DMR initialisation — include before setRootCase.H.
// For multi-region (foamMultiRun), define DMR_ALL_REGIONS true first.
//
// Runtime control (OpenFOAM-side hooks):
//   FOAM_DMR_ENABLED=1       activates DMR at runtime (required even when
//                            the binary was compiled with -DFOAM_USE_DMR)
//   FOAM_DMR_LOG=1           write DMR-internal subprocess and lifecycle
//                            output to <case>/log.dmr/ (default on; set
//                            FOAM_DMR_LOG=0 to silence)
//
// DMR-library knobs (DMR_DEFAULT_INHIBITOR for reconfiguration cadence,
// DMR_DEFAULT_POLICY_MIN/MAX, DMR_PRINT_ANALYTICS, DMR_DEBUG_LEVEL,
// DMR_PROCS_PER_NODE, ...) are read directly by the DMR library and
// need no OpenFOAM-side bridging.

#ifndef DMR_ALL_REGIONS
#define DMR_ALL_REGIONS false
#endif

// Runtime toggle: compiled with -DFOAM_USE_DMR means DMR is active by default.
// Set FOAM_DMR_ENABLED=0 to disable at runtime (e.g. comparison runs).
const bool dmrEnabled = [](){
    const char* env = std::getenv("FOAM_DMR_ENABLED");
    return !env || std::string(env) != "0";
}();

const bool dmrAllRegions = DMR_ALL_REGIONS;

// Parse -case from argv up-front (OpenFOAM's argList is not yet built)
// so both the dynamic-mesh probe below and the DMR_AUTO restart callback
// can reference it.  Defaults to "." when omitted.
const std::string dmrCasePath = [&](){
    for (int i = 1; i < argc - 1; ++i)
    {
        if (std::string(argv[i]) == "-case") return std::string(argv[i + 1]);
    }
    return std::string(".");
}();

// Pin DMR reconfiguration to writeInterval boundaries when the case has
// any dynamic mesh.  Cases with cyclicAMI / nonConformalProcessorCyclic
// patches can hit zero-overlap geometry where decomposePar's from-scratch
// stitcher computes 0/0 weights and crashes.  Aligning with writeInterval
// gives us a state the case author already vetted as fully consistent.
// Static-mesh cases are unaffected and reconfigure on any step.
const bool dmrDynamicMesh = Foam::dmrHasDynamicMesh(dmrCasePath);

// On dynamic-mesh cases we throttle dmr_check() with our own OpenFOAM-side
// counter ticked at runTime.writeTime().  We still want the user-facing
// semantics of DMR_DEFAULT_INHIBITOR to be uniform ("skip N opportunities
// before checking"), so we read the value once here and apply it to our
// counter.  See dmrCheck.H for the gating logic.
const int dmrDynInhibitor = [](){
    const char* env = std::getenv("DMR_DEFAULT_INHIBITOR");
    return env ? std::atoi(env) : 0;
}();

// Reconfiguration policy.  ROUND_POLICY deterministically cycles through
// N_min..N_max by doubling, then resets — used for debugging and validation
// because every reconfiguration path is exercised.  Switch to CE_POLICY
// (communication-efficiency driven) for production runs; the choice is a
// build-time decision, not a user-facing option.
const DMRSuggestion dmrPolicy = ROUND_POLICY;

if (dmrEnabled)
{
    // Initialise MPI before DMR.
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    }

    if (dmrDynamicMesh)
    {
        Info<< "DMR: dynamic mesh detected. Reconfiguration is pinned"
            << " to writeInterval boundaries; an OpenFOAM-side counter"
            << " honours DMR_DEFAULT_INHIBITOR=" << dmrDynInhibitor
            << " (skip N writeIntervals between checks)." << nl;
    }

    // DMR_AUTO phases: init / running / restart / cleanup.
    // On restart, dmrRestart() updates the case dicts and decomposes for the
    // new process count — safe to run inside DMR_AUTO.
    DMR_AUTO(
        dmr_init(argc, argv),
        (void)NULL,
        Foam::dmrRestart(dmrCasePath, dmrAllRegions),
        (void)NULL
    );

    // For dynamic-mesh cases, throttling is done by an OpenFOAM-side
    // counter ticked at writeTime (see dmrCheck.H).  Disable DMR's own
    // inhibitor so each gated dmr_check() actually evaluates the policy
    // instead of being silently skipped by DMR's internal counter.
    if (dmrDynamicMesh)
    {
        dmr_set_reconf_step_inhibitor(0);
    }
}
