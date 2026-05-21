// DMR initialisation — include before setRootCase.H.
// For multi-region (foamMultiRun), define DMR_ALL_REGIONS true first.
//
// DMR is opt-in: pass -dmr on the command line to activate. The
// dmr_wrapper launcher appends this flag automatically.

#ifndef DMR_ALL_REGIONS
#define DMR_ALL_REGIONS false
#endif

Foam::argList::addBoolOption
(
    "dmr",
    "Enable DMR malleability (requires launch via dmr_wrapper)."
);

// argList is not built yet; scan argv directly.
const bool dmrEnabled = [&](){
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "-dmr") return true;
    }
    return false;
}();

const bool dmrAllRegions = DMR_ALL_REGIONS;

const std::string dmrCasePath = [&](){
    for (int i = 1; i < argc - 1; ++i)
    {
        if (std::string(argv[i]) == "-case") return std::string(argv[i + 1]);
    }
    return std::string(".");
}();

// ROUND_POLICY for debugging (deterministic doubling cycle); switch to
// CE_POLICY for production. Build-time choice, not user-facing.
const DMRSuggestion dmrPolicy = ROUND_POLICY;

if (dmrEnabled)
{
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    }

    DMR_AUTO(
        dmr_init(argc, argv),
        (void)NULL,
        Foam::dmrRestart(dmrCasePath, dmrAllRegions),
        (void)NULL
    );
}
