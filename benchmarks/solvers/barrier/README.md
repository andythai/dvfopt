# Barrier Solver Benchmarks

Benchmarks for the penalty → log-barrier L-BFGS-B solver
(`iterative_3d_barrier` and the 2D / GPU variants).

The barrier solver is a two-phase unconstrained optimiser:

1. **Phase 1 — exterior quadratic penalty.** Smooth everywhere, so it
   can start from deeply infeasible fields (large negative Jdet) that
   make the log-barrier undefined. λ-continuation drives `min J ≥
   threshold + margin`.
2. **Phase 2 — log-barrier interior point.** Maintains strict
   feasibility via a line search that rejects steps crossing the
   barrier. μ-continuation polishes the solution close to the feasible
   boundary.

Both phases run under `scipy.optimize.minimize(method="L-BFGS-B")`
(numpy backend) or `torch.optim.LBFGS` (GPU backend), reusing the same
analytical `dJ/du` sparse Jacobian as the SLSQP solver.

## Notebooks

### [benchmark-3d-barrier.ipynb](benchmark-3d-barrier.ipynb)
Runs `iterative_3d_barrier` on the same test fields used by the SLSQP
3D benchmark: synthetic random DVFs (5³, 4×6×5, 6×8×8, 8×8×8) and the
real Elastix volume at three downsample factors (1/4, 1/2, 1/1).
Reports neg-Jdet count, min Jdet, L2 distortion, and wall time. Uses
the **windowed** mode so even the full `(528, 320, 456)` volume is
tractable on CPU.

### [benchmark-3d-barrier-windowed-vs-fullgrid.ipynb](benchmark-3d-barrier-windowed-vs-fullgrid.ipynb)
Compares `iterative_3d_barrier(windowed=True)` against `windowed=False`
across multiple downsample factors. Includes a background-thread RSS
sampler (`_peak_rss_during`) to record peak memory per run and
automatically skips the full-grid mode above a DOF cap
(`FULLGRID_MAX_DOFS = 60_000_000`, ~10 GB working memory). Produces
side-by-side bar plots for wall time, peak memory, and L2 distortion.

### [benchmark-barrier-cpu-vs-gpu.ipynb](benchmark-barrier-cpu-vs-gpu.ipynb)
Compares the numpy/scipy L-BFGS-B (CPU) and torch L-BFGS (GPU)
backends of the barrier solver across 2D and 3D test cases. GPU has a
fixed launch overhead (~1–2 s per L-BFGS step), so small problems lose
on GPU; the crossover sits around 10k–100k variables on typical
hardware.

## Output

All notebooks write to `output/barrier/<notebook_name>/` (or
`output/barrier-gpu/...` for the GPU-specific notebook) relative to
the repo root. See [`../../README.md`](../../README.md) for the shared
output convention.
