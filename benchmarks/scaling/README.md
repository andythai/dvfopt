# Scaling Benchmarks

Measure how the correction methods scale with grid size, folding
severity, and initial Jdet magnitude. Each metric has two variants:
one for the SLSQP solver and a `-barrier` companion for the penalty →
log-barrier solver.

## Notebooks

### [benchmark-scalability.ipynb](benchmark-scalability.ipynb) · [benchmark-scalability-barrier.ipynb](benchmark-scalability-barrier.ipynb)
Measures runtime, L2 error, and iteration count as the grid resolution
grows. A fixed random DVF pattern is generated at a small base size
and upscaled to each target resolution so the folding pattern stays
consistent across sizes.

Sizes tested: **5×5, 8×8, 10×10, 12×12, 15×15, 18×18, 20×20**.

### [benchmark-folding-severity.ipynb](benchmark-folding-severity.ipynb) · [benchmark-folding-severity-barrier.ipynb](benchmark-folding-severity-barrier.ipynb)
Generates DVFs with increasing folding by (a) scaling displacement
**magnitude** from mild to extreme and (b) increasing the **density**
of folding sources. Tracks initial neg-Jdet count, runtime, final L2
error, and convergence.

### [benchmark-l2-jdet-correlation.ipynb](benchmark-l2-jdet-correlation.ipynb) · [benchmark-l2-jdet-correlation-barrier.ipynb](benchmark-l2-jdet-correlation-barrier.ipynb)
Per-pixel correlation between initial Jacobian determinant and
correction magnitude. The SLSQP version sweeps all four constraint
modes; the barrier version uses only the default Jdet constraint.
Expected behaviour: pixels with strongly negative initial Jdet require
the largest displacements to fix, yielding a strong negative
correlation.

## Output

SLSQP variants write to `output/slsqp/<notebook_name>/`; barrier
variants write to `output/barrier/<notebook_name>/`. See
[`../README.md`](../README.md) for the shared output convention.
