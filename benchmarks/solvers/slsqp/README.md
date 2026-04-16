# SLSQP Solver Benchmarks

Benchmarks for the SciPy SLSQP-based correction methods
(`iterative_serial`, `iterative_parallel`, `iterative_3d`).

## Notebooks

### [benchmark-serial-vs-parallel.ipynb](benchmark-serial-vs-parallel.ipynb)
Compares the serial `iterative_serial` against the parallelised
`iterative_parallel` (non-overlapping windows batched through
`ProcessPoolExecutor`). Reports wall time, L2 distortion, and
convergence on identical 2D test fields.

### [benchmark-constraint-modes.ipynb](benchmark-constraint-modes.ipynb)
Compares the four 2D constraint configurations:

| Mode | Flags |
|------|-------|
| Jacobian only | default |
| Jacobian + Shoelace | `enforce_shoelace=True` |
| Jacobian + Injectivity | `enforce_injectivity=True` |
| All constraints | both `True` |

Metrics: runtime, L2 error, final min Jdet, outer SLSQP iterations, and
whether all negatives were eliminated.

### [benchmark-windowed-vs-fullgrid.ipynb](benchmark-windowed-vs-fullgrid.ipynb)
Compares the windowed iterative approach (worst pixel → bounding box →
frozen-edge SLSQP → repeat) against a single-shot full-grid SLSQP over
all `2*H*W` variables. Demonstrates why windowing is necessary above
trivial grid sizes.

### [benchmark-3d-correction.ipynb](benchmark-3d-correction.ipynb)
Exercises `iterative_3d` on synthetic random DVFs (5×5×5, 6×8×8, 8×8×8)
and real downsampled Elastix volumes. Includes 3D Jdet scatter plots,
per-voxel highlighting, deformed-grid views, and per-slice Jdet
heatmaps.

## Output

All notebooks write to `output/slsqp/<notebook_name>/` relative to the
repo root. See [`../../README.md`](../../README.md) for the shared
output convention.
