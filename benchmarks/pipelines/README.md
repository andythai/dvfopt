# Pipeline Benchmarks

End-to-end slice-wise correction pipelines. Each notebook takes a 3D
displacement field, extracts axial slices, corrects folding slice by
slice, and recomposes the corrected 3D field.

Two input modes are supported by each notebook:

1. **Synthetic test case** (default) — a small 3D warp volume with
   deliberate folding in selected slices. No external data required.
2. **ANTs / NIfTI warp** — load a real `.nii.gz` displacement field
   from ANTs registration and correct slices with negative Jacobians.

## Pipeline per slice

1. Extract one axial slice → convert to `(3, 1, H, W)`.
2. Compute initial Jacobian determinant; skip if already fold-free.
3. Run the chosen iterative SLSQP variant.
4. Record runtime, L2 distortion, and final Jdet statistics.
5. Reassemble the corrected slice into the output volume.

## Notebooks

### [correct-3d-slices.ipynb](correct-3d-slices.ipynb)
Baseline version using the default serial iterative SLSQP with the
Jacobian-only constraint.

### [correct-3d-slices-serial.ipynb](correct-3d-slices-serial.ipynb)
Explicit serial variant — exercises `iterative_serial` on every
slice sequentially. Useful as the comparison baseline for the parallel
and multi-constraint variants.

### [correct-3d-slices-parallel.ipynb](correct-3d-slices-parallel.ipynb)
Uses `iterative_parallel`, which batches non-overlapping sub-windows
into a `ProcessPoolExecutor`. On Windows, falls back to serial when a
slice only needs one window (avoids spawn overhead).

### [correct-3d-slices-multi-constraint.ipynb](correct-3d-slices-multi-constraint.ipynb)
Runs the same synthetic 3D warp under four constraint configurations
and compares the results using the injectivity checks from
`test-global-folding.ipynb`:

| Mode | `enforce_shoelace` | `enforce_injectivity` |
|------|---------------------|-----------------------|
| jdet-only | False | False |
| jdet+shoe | True | False |
| jdet+inject | False | True |
| jdet+shoe+inject | True | True |

## Output

All notebooks write to `output/slsqp/<notebook_name>/` relative to the
repo root. See [`../README.md`](../README.md) for the shared output
convention.
