// DMR initialisation — include before setRootCase.H.
// For multi-region (foamMultiRun), define DMR_ALL_REGIONS true first.
//
// Runtime control:
//   FOAM_DMR_ENABLED=1       activates DMR (required even when compiled in)
//   DMR_CHECK_INTERVAL       steps between reconfiguration checks
//   DMR_DEFAULT_POLICY_MIN   node lower bound
//   DMR_DEFAULT_POLICY_MAX   node upper bound

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
    // new process count, all via subprocess — safe to run inside DMR_AUTO.
    DMR_AUTO(
        dmr_init(argc, argv),
        (void)NULL,
        Foam::dmrRestart(dmrCasePath, dmrAllRegions),
        (void)NULL
    );

    // Bridge environment variables to the DMR C API.
    // The Docker DMR library (v2.0.0) does not read these natively.
    {
        const char* envMax = std::getenv("DMR_DEFAULT_POLICY_MAX");
        if (envMax) dmr_set_policy_max_nodes(std::atoi(envMax));

        const char* envMin = std::getenv("DMR_DEFAULT_POLICY_MIN");
        if (envMin) dmr_set_policy_min_nodes(std::atoi(envMin));
    }
}
