// DMR initialisation — include before setRootCase.H.
// For multi-region (foamMultiRun), define DMR_ALL_REGIONS true first.
//
// Runtime control:
//   FOAM_DMR_ENABLED=1   activates DMR (required even if compiled with DMR)
//   DMR_CHECK_INTERVAL   how many time steps between reconfiguration checks
//   DMR_DEFAULT_POLICY_MAX / MIN   node limits for the DMR policy

#ifndef DMR_ALL_REGIONS
#define DMR_ALL_REGIONS false
#endif

// Runtime toggle: compiled with -DFOAM_USE_DMR means DMR is active
// by default. Set FOAM_DMR_ENABLED=0 to disable (e.g. comparison runs).
const bool dmrEnabled = [](){
    const char* env = std::getenv("FOAM_DMR_ENABLED");
    return !env || std::string(env) != "0";
}();

const bool dmrAllRegions = DMR_ALL_REGIONS;

if (dmrEnabled)
{
    // Initialise MPI before DMR
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    }

    // Parse -case from argv (OpenFOAM's argList is not constructed yet)
    std::string dmrCasePath = ".";
    for (int i = 1; i < argc - 1; ++i)
    {
        if (std::string(argv[i]) == "-case")
        {
            dmrCasePath = argv[i + 1];
            break;
        }
    }

    // DMR_AUTO: init DMR, on restart decompose for new process count
    DMR_AUTO(
        dmr_init(argc, argv),
        (void)NULL,
        dmrRestart(dmrCasePath, dmrAllRegions),
        (void)NULL
    );

    // Bridge environment variables to DMR C API.
    // The Docker DMR library (v2.0.0) does not read these natively.
    {
        const char* envMax = std::getenv("DMR_DEFAULT_POLICY_MAX");
        if (envMax) dmr_set_policy_max_nodes(std::atoi(envMax));

        const char* envMin = std::getenv("DMR_DEFAULT_POLICY_MIN");
        if (envMin) dmr_set_policy_min_nodes(std::atoi(envMin));
    }

    // Phase 2 restart: in-process decompose for the new process count.
    // Runs AFTER DMR_AUTO so DMR_INTERCOMM has already been freed by
    // dmr_reconfigure().  MPI_COMM_WORLD is clean at this point.
    // No-op on fresh starts (dmrRestartPending_ is false).
    Foam::dmrDecomposeInProcess(dmrCasePath, dmrAllRegions);
}
