// DMR grow-side redistribute — include between createTime.H and mesh
// creation. After a GROW reconfiguration the checkpoint exists only in
// the old, smaller processor layout; this hook redistributes it to the
// current world size in-process (parallel, via fvMeshDistribute),
// replacing the serial decomposePar. It runs here because OpenFOAM's
// parallel layer is up (argList initialised Pstream) and the mesh has
// not been read yet. No-op unless this process generation follows a
// DMR GROW reconfiguration.

if (dmrEnabled)
{
    Foam::dmrGrowRedistribute(runTime, dmrAllRegions);
}
