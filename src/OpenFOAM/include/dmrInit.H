// DMR initialisation — include before setRootCase.H.
// For multi-region (foamMultiRun), define DMR_ALL_REGIONS true first.
//
// Runtime control (OpenFOAM-side hooks):
//   FOAM_DMR_ENABLED=1       activates DMR (required even when compiled in)
//   FOAM_DMR_LOG=1           write DMR-internal subprocess output to
//                            <case>/log.dmr/ (default on; set 0 to silence)
//   DMR_CHECK_INTERVAL       steps between reconfiguration checks
//
// DMR-library knobs (DMR_DEFAULT_POLICY_MIN, DMR_DEFAULT_POLICY_MAX,
// DMR_PRINT_ANALYTICS, DMR_DEBUG_LEVEL, DMR_PROCS_PER_NODE, ...) are read
// directly by the DMR library and need no OpenFOAM-side bridging.

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

    // Parse -case from argv (OpenFOAM's argList is not constructed yet).
    std::string dmrCasePath = ".";
    for (int i = 1; i < argc - 1; ++i)
    {
        if (std::string(argv[i]) == "-case")
        {
            dmrCasePath = argv[i + 1];
            break;
        }
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
}
