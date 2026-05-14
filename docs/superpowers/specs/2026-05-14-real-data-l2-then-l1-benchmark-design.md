# Real-data L2-then-L1 benchmark — 2D per-slice and 3D full-volume

**Status**: design · **Date**: 2026-05-14

## Goal

Apply the manuscript's `L2-multipass → L1-polish` correction methodology (established in notebooks 17 and 18) to the full real registration field at `data/corrected_correspondences_count_touching/registered_output/deformation3d.npy` (shape `(3, 528, 320, 456)`), and record the per-case metrics needed for the manuscript's empirical sections.

Two notebooks, one per regime:

- `notebooks/manuscript/03_benchmark_2d_real_full.ipynb` — runs the 2D 2-triangle correction on **every** z-slice of the field.
- `notebooks/manuscript/04_benchmark_3d_real_full.ipynb` — runs a 3D 6-tet correction on the full volume.

Both notebooks use the same `L2 → L1` two-stage recipe, save incremental results, and produce manuscript-ready summary plots.

## Data conventions

- The field's channel 0 (dz) is identically zero — it's a stack of 2D registrations. So each z-slice is a genuinely 2D problem, and the 3D run treats dz=0 as the initial condition the solver is free to modify.
- Convention follows CLAUDE.md: `(3, D, H, W)` with channels `[dz, dy, dx]`; threshold `τ = 0.01`; smoothed-L1 epsilon `ε = 1e-4`.

## Solver choices (settled by user)

| Decision | Choice |
|---|---|
| Solver | Windowed iterative + L1 polish (full-grid SLSQP doesn't scale to 320×456) |
| 2D scope | All 528 slices |
| 3D scope | Full volume (windowed) |
| Time budget | Aggressive — drive every case to feasibility, no strict per-slice cap |
| Output | `notebooks/manuscript/output/{2d_real_full,3d_real_full}/` |

## Notebook 1 — `03_benchmark_2d_real_full.ipynb`

### Per-slice pipeline

For each `z ∈ [0, 528)`:

1. **Extract slice.** `phi_z = deformation3d[:, z:z+1, :, :].copy()` (shape `(3, 1, H, W)`, the existing `iterative_serial` signature).
2. **Initial check.** Compute `T1, T2 = triangle_sign_areas2D(phi_z[1:, 0])`; record initial `n_neg_tri = int((T1 <= 0).sum() + (T2 <= 0).sum())` and `min_tri = min(T1.min(), T2.min())`.
3. **Skip if clean.** If `n_neg_tri == 0`, record a "no work needed" row and move on.
4. **L2 phase.** Run the existing windowed solver:
   ```python
   from dvfopt.core.iterative import iterative_serial
   phi_l2 = iterative_serial(
       phi_z, enforce_triangles=True,
       max_iterations=20000, err_tol=1e-5, verbose=0,
   )
   ```
   The solver loops over worst-pixel windows until `n_neg_tri = 0`. The `enforce_triangles=True` flag uses the 2-triangle constraint (the manuscript's check) instead of central-diff Jdet.
5. **L1 polish phase.** Identify cells where the L2 phase actually moved the field:
   - `touched = |phi_l2 - phi_z| > 1e-3` (per-pixel, max across dy/dx).
   - Use `scipy.ndimage.label` to get connected components.
   - For each component: build its bounding box + 1 px positive border (matching the windowed-SLSQP pattern in `dvfopt.core.slsqp.iterative`), run a single SLSQP call with the smoothed-L1 objective `Σ √(Δ² + ε²)`, anchor at the **original** `phi_z` slice (not the L2 result), edges frozen, 2-triangle constraint.
   - After each component-solve, verify `n_neg_tri == 0` on the *whole* slice; if a new fold appeared (unlikely with frozen edges but possible if a window touches another component), reject the polish for that component and keep the L2 solution there.
6. **Record row.** Append to `output/2d_real_full/per_slice.csv` with columns:
   ```
   z, H, W,
   init_n_neg_tri, init_min_tri,
   L2_n_neg_tri, L2_min_tri, L2_L1, L2_L2, L2_iter, L2_t,
   L1_L1, L1_L2, L1_min_tri, L1_n_neg_tri, L1_components, L1_iter_total, L1_t,
   total_t, feasible (= L2_n_neg_tri == 0 at end),
   l1_drop_pct (= 100 * (L2_L1 - L1_L1) / L2_L1 when feasible)
   ```
7. **Checkpoint.** Every N=20 slices, snapshot `phi_corrected_so_far[:, :z+1]` to `output/2d_real_full/checkpoint.npz` for resume.

### Resumability

On startup the notebook reads `per_slice.csv` and skips any `z` already present. The checkpoint `.npz` is reloaded so subsequent slices append to the same corrected volume.

### Aggregate analysis

After all slices done:

- **Headline stats** — count of slices that needed work, count that reached feasibility, total wall time, distribution of `L1_drop_pct`.
- **Plots** (saved as PDF/PNG to `output/2d_real_full/`):
  1. `per_slice_fold_overview.{pdf,png}` — line plot of `init_n_neg_tri` vs `z`, with `L1_n_neg_tri` overlaid (should be ~0 everywhere feasible).
  2. `l1_vs_l2_sparsity.{pdf,png}` — scatter of `L2_L1` vs `L1_L1` per slice, with the y=x diagonal; cluster on the L1<L2 side confirms the polish helps.
  3. `time_distribution.{pdf,png}` — histogram of per-slice runtimes split by L2 vs L1 phase.
  4. `worst_slice_visual.{pdf,png}` — pick the slice with the highest initial `n_neg_tri`, show its BEFORE / L2 / L1 deformation grids + residual heatmap (mirroring notebook 14's style).

## Notebook 2 — `04_benchmark_3d_real_full.ipynb`

### Custom windowed 3D-tet solver

Inline in the notebook (the manuscript's tet-volume constraint must be what's actually enforced; `iterative_3d` in the package uses J_CD, which is a different metric).

Building blocks:

```python
# Reuse the tet helpers from notebooks 17/18:
CUBE_CORNERS, TET_INDICES, warp_corners, tet_signed_volumes, pack_phi, unpack_phi
```

Outer loop (analogous to `iterative_3d`'s pattern but with the tet-volume constraint):

```
while n_neg_tet > 0 and outer_iter < max_outer:
    # 1. Find worst tet -> its cell coords.
    V = tet_signed_volumes(phi)
    (ti, cz, cy, cx) = argmin_unraveled(V)
    # 2. Identify the connected component of "folded cells"
    #    (cells whose min-tet is <= threshold) that contains (cz, cy, cx).
    component_mask = label_3d(V.min(axis=0) <= threshold)[component_of((cz,cy,cx))]
    # 3. Bounding box + 1-voxel positive border, min size (3,3,3).
    (z0,z1, y0,y1, x0,x1) = bbox_with_border(component_mask)
    # 4. Run SLSQP on that sub-window:
    #    - Variables: phi values at interior voxels.
    #    - Edges frozen at current values.
    #    - Constraint: V_t >= threshold for each tet whose 8 corners
    #      lie inside the (closed) sub-window.
    #    - Objective: L2 (objective_euc), anchored at original phi.
    phi_local_new = slsqp_solve_local_l2(phi, (z0,z1, y0,y1, x0,x1), phi_anchor, ...)
    phi[..., z0:z1, y0:y1, x0:x1] = phi_local_new
    # 5. Re-check; if no progress (worst tet unchanged), grow the window.
```

Window growth follows the same "double until progress" pattern as the 2D `iterative_serial`.

### L1 polish phase

After the L2 loop reaches `n_neg_tet = 0`:

1. Identify *touched* voxels (where `|phi_l2 - phi_initial| > 1e-3`) and their connected components.
2. For each component: bounding-box sub-window, edges frozen, SLSQP with smoothed-L1 objective + tet-volume constraint, anchored at the **original** `phi_initial`.
3. Accept the polish for each component only if it strictly lowers `L1` on that window *and* keeps `n_neg_tet = 0` globally.

### Trajectory recording

One CSV row per outer iteration (`L2_pass_k` and `L1_polish_component_k`):

```
stage, iter_idx, window_z0, window_z1, window_y0, window_y1, window_x0, window_x1,
window_voxels, slsqp_nit, slsqp_success,
n_neg_tet_before, n_neg_tet_after,
min_tet_before, min_tet_after,
L1_before, L1_after, L2_before, L2_after,
t_seconds
```

Plus the per-iteration corrected-field snapshots at checkpoint intervals.

### Aggregate analysis

- Headline numbers — initial `n_neg_tet`, final `n_neg_tet`, total iterations, total wall time, final `L1`, final `L2`.
- **Plots**:
  1. `fold_count_trajectory.{pdf,png}` — `n_neg_tet` vs outer-iteration index (L2 phase) with L1-polish phase shaded.
  2. `l1_l2_norm_trajectory.{pdf,png}` — `L1` and `L2` norms vs iteration on twin axes.
  3. `fold_spatial_distribution.{pdf,png}` — initial fold-cell locations projected to each axis (xy, xz, yz) so the manuscript can show where in the volume the corrections happened.

## Run order

User confirmed: build both notebooks, then execute the 2D notebook first, then the 3D. Both are run as background processes (papermill or `jupyter nbconvert --execute`) since the aggressive budget means tens of hours of compute. Per-slice CSV append guarantees no work is lost on kernel kill.

## Out of scope

- Comparing to other correction methods (Elastix, VoxelMorph, etc.) — those benchmarks already live in `benchmarks/registration/`.
- Modifying the package's `iterative_3d` to support the tet-volume constraint. The custom windowed solver lives inline in the manuscript notebook so the reader sees exactly what was run.
- Parallel windowed solving — serial only, matching notebook 18's setup.
- Speed/quality trade-off plots — covered by the existing `benchmarks/solvers/` notebooks.

## Open questions

None blocking. The user has chosen all recommended options including the aggressive budget.
