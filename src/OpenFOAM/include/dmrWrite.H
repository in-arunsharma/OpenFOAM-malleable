// DMR natural write — replaces runTime.write() in the time loop.
// Identical to runTime.write() unless the uniform-output path is
// active (FOAM_DMR_UNIFORM_WRITE=1 and the run is above the MIN
// layout). A malleable run changes its process count as it goes, so
// writing each output time in the layout of the moment leaves the
// series decomposed differently from one write to the next, which is
// not readable as a case. When the path is active the fields are left
// out of this write and handed to a helper world spawned for the
// purpose, which re-emits them in the MIN layout while the solver
// carries on; everything else Time writes (uniform/time in
// particular) still goes out here.

if (dmrEnabled)
{
    Foam::dmrWrite(runTime, dmrAllRegions);
}
else
{
    runTime.write();
}
