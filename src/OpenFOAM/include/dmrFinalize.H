// DMR clean shutdown — include after the time loop.
// Calls dmr_finalize() via DMR_AUTO, then merges any remaining processor
// directories into the serial case and sweeps DMR-forced checkpoint sentinels.

if (dmrEnabled)
{
    DMR_AUTO(dmr_finalize(), (void)NULL, (void)NULL, (void)NULL);

    const std::string caseStr(runTime.globalPath().c_str());
    Foam::dmrFinalize(caseStr, dmrAllRegions);
}
