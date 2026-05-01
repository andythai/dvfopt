# 3D Tetrahedral Signed-Volume Check (extension of 2-triangle check)

**Status**: design · **Date**: 2026-05-01

## Goal

Extend the 2D 2-triangle sign check (`dvfopt.jacobian.triangle_sign`) to 3D by splitting each voxel cell into 6 tetrahedra along one body diagonal. Use the resulting per-tet signed volumes for two purposes:

1. **Detection** — catch local 3D folds (bowtie analogs) that central-difference Jdet misses.
2. **Correction** — drive SLSQP with tet-volume constraints `≥ THRESHOLD` instead of (or in addition to) the existing central-diff Jdet constraint.

Deliverable: a single new exploration notebook `notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb` that demonstrates and visualises both. No package code changes are required for this exploration; helpers live in the notebook.

## Geometry — 6-tet decomposition along a body diagonal

A voxel cell has 8 corners. Index them by their `(z, y, x)` offset from the cell's origin:

```
v0 = (0,0,0)  v1 = (0,0,1)  v2 = (0,1,0)  v3 = (0,1,1)
v4 = (1,0,0)  v5 = (1,0,1)  v6 = (1,1,0)  v7 = (1,1,1)
```

Pick the body diagonal `v0 → v7`. The 6 tets correspond to the 6 monotone paths from `v0` to `v7` along cube edges (one bit flip per step — there are `3! = 6` such paths). Each tet is `(v0, A, B, v7)` where `A` and `B` are the two intermediate corners on the path. All 6 tets share the body diagonal `v0–v7` as an edge.

```
T0 = (v0, v1, v3, v7)   # path x, y, z   (x first)
T1 = (v0, v1, v5, v7)   # path x, z, y
T2 = (v0, v2, v3, v7)   # path y, x, z   (y first)
T3 = (v0, v2, v6, v7)   # path y, z, x
T4 = (v0, v4, v5, v7)   # path z, x, y   (z first)
T5 = (v0, v4, v6, v7)   # path z, y, x
```

Signed volume of a tetrahedron with vertices `(a, b, c, d)` in `(x, y, z)` coordinates:

```
V = (1/6) · ( (b - a) · ((c - a) × (d - a)) )
```

We assemble vertex coordinates from the warped grid:

```
warp[z, y, x] = (x + dx[z,y,x], y + dy[z,y,x], z + dz[z,y,x])
```

Output shape mirrors `triangle_sign_areas2D`: `(6, D-1, H-1, W-1)` per-tet signed volumes per cell. Sign convention: positive = valid, zero = degenerate, negative = flip.

The orientation of each tet's `(A, B)` pair determines the sign of the unwarped volume. Half the 6 path orderings above are even permutations and half are odd; we apply a per-tet sign flip (a fixed `±1` vector of length 6) so that every tet's unwarped volume is `+1/6`. The flip pattern is computed once at notebook import and verified by asserting `min(V) > 0` on the identity field.

This is the structural analog of the 2D 2-triangle check: a single shared diagonal of the cell, with the minimum-by-shared-diagonal number of simplices covering it (2 in 2D, 6 in 3D).

## Test cases

Construct minimal synthetic deformations on small grids:

1. **3D bowtie (the canonical one).** `D = H = W = 7`. Zero everywhere except `dx[3, 3, 3] = +1.2`, `dx[3, 3, 4] = −1.2`. Direct lift of the 2D bowtie from `01_vs-central-diff.ipynb` along the x-axis. Expectation: central-diff Jdet stays positive; at least one tet in cells `(3, 3, 3)` and `(3, 3, 4)` flips.

2. **z-axis bowtie.** Same magnitude swap on `dz[3, 3, 3] = +1.2`, `dz[4, 3, 3] = −1.2`. Validates the check is rotation-equivariant and the body-diagonal choice doesn't bias detection.

3. **xy diagonal swap.** `dx[3,3,3]=+0.8, dy[3,3,3]=+0.8, dx[3,4,3]=-0.8, dy[3,4,3]=-0.8`. Multi-axis fold to stress the constraint.

4. **(Optional) 3D analog of an existing `test_cases` fold.** If quick to wire up: lift `01a_10x10_crossing` to a thin 3-slice 3D field by zeroing `dz` and stacking. Sanity check against the 2D triangle counts.

For each case, report:
- `n_neg` from central-diff Jdet (`jacobian_det3D`)
- `n_neg` from min-tet sign across all 6 tets per cell
- per-tet flip counts (which of T0..T5 flips)

## SLSQP correction comparison

Two solver runs per case:

- **S-CD**: full-grid SLSQP with central-diff Jdet ≥ THRESHOLD constraint (existing approach in `dvfopt/core/solver3d.py::_full_grid_step_3d`).
- **S-TET**: full-grid SLSQP with all 6 per-cell tet signed volumes ≥ THRESHOLD as the constraint. Objective stays L2 (`objective_euc`).

The tet constraint vector for a cell of shape `(D, H, W)` has length `6 · (D-1) · (H-1) · (W-1)`. Variable packing follows the existing 3D solver convention `[dx_flat, dy_flat, dz_flat]`.

Hypothesis to confirm: on the 3D bowtie, S-CD will report success with all-positive Jdet but the tet check still flags folds (same pathology as 2D). S-TET will produce a feasible field under both checks. Report final L2/L1 cost, min Jdet (CD), min tet volume, and iteration count for each.

## Visualisation

Per case, a multi-panel matplotlib figure:

1. **Mid-z slice — central-diff Jdet heatmap** (`RdBu_r`).
2. **Mid-z slice — min-tet signed volume per cell** (`RdBu_r`, shared scale).
3. **3D warped grid view** (mplot3d): faintly draw the reference cube grid; overlay warped edges in blue; outline cells where any tet has volume `≤ 0` in dark red.
4. **Per-tet flip count bar chart** across T0..T5 to surface which tet types are most sensitive in each fold geometry.

The 3D plot is intentionally limited to the small synthetic grids (≤ 7³). Full-volume 3D plotting is out of scope for this notebook.

## Module-level code

Stay inline in the notebook for this exploration. If the approach pans out, a follow-up PR can add a `dvfopt/jacobian/tetrahedron_sign.py` module mirroring `triangle_sign.py` with `tetrahedron_sign_det3D`, `tetrahedron_sign_volumes3D`, `tetrahedron_sign_constraint`. That module work is **explicitly not in scope** for this notebook — the goal is to validate the geometry and the SLSQP integration before promoting helpers into the package.

## Notebook structure

`notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb`:

1. **Intro** — problem statement, link back to `01_vs-central-diff.ipynb`, definition of the 6-tet decomposition with diagram.
2. **Helpers** — `_warp_corners`, `_tet_signed_volumes`, `tetrahedron_sign_volumes3D`, `tetrahedron_sign_constraint`, `_count_negative_tets`. Sanity check on identity field returns all positive volumes.
3. **Test case 1: 3D x-axis bowtie** — build, central-diff vs tet diagnostics, viz.
4. **Test case 2: z-axis bowtie** — same, demonstrates axis-equivariance.
5. **Test case 3: xy diagonal swap** — multi-axis fold.
6. **SLSQP correction** — S-CD vs S-TET on case 1, table of metrics, residual viz.
7. **Summary** — when the tet check disagrees with central-diff in 3D, mirroring the 2D summary in `01_vs-central-diff.ipynb`.

## Out of scope

- Promoting helpers into `dvfopt/jacobian/` (follow-up).
- Adding tet-volume constraint to `iterative_3d` (follow-up).
- 24-tet Kuhn split (future work analog to `triangle_det2D`).
- 5-tet chirality variants (alternative; not pursued).
- Full-volume 3D rendering of large grids.
- Performance benchmarking (this is exploratory, not a benchmark notebook).

## Success criteria

1. Notebook runs end-to-end on a fresh kernel without errors.
2. Identity-field sanity check returns all-positive tet volumes.
3. On the 3D x-axis bowtie, central-diff Jdet reports `n_neg = 0` while the tet check reports `≥ 1` cell with a flipped tet.
4. S-TET produces a field with `min_tet_volume ≥ THRESHOLD` on at least one of the 3 test cases where S-CD fails to.
5. 3D viz renders without performance issues on a 7³ grid.
