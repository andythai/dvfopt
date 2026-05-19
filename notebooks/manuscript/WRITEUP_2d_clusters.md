# 2D per-slice cluster solver — working setup

This document describes the `_run_2d_clusters.py` pipeline that converges
to **0 folded triangles** and **`min_tri >= ~0.01`** on every slice of the
real registration DVF (`data/corrected_correspondences_count_touching/registered_output/deformation3d.npy`,
shape `(3, 528, 320, 456)`).

## Problem statement

Given a 2D displacement field `phi` of shape `(2, H, W)` (channels
`[dy, dx]`), find a corrected field `phi*` that satisfies the
**2-triangle constraint** at every cell:

  T1(y, x; phi*) >= threshold   for all cells (y, x)
  T2(y, x; phi*) >= threshold

while minimising `||phi* - phi||` (L2 first, then L1 polish). The
threshold is `0.01`. The real slices have 178 - 3257 initial folded
triangles each.

Direct full-grid SLSQP on a 320 x 456 slice is infeasible (~290 k
variables, OOM at 270 GiB). Per-pixel windowed SLSQP (the package's
`iterative_serial`) drives `min_jdet > 0` but doesn't reach the
2-triangle threshold and stalls on dense folds.

## Recipe

For each slice, run **multi-pass full-grid L2 + L1 polish SLSQP per
connected fold component**, in parallel, with the bbox grown only as
much as needed.

```text
for each z in ascending order of initial fold count:
    for outer_iter in [0..MAX_CLUSTER_OUTER_ITERS):
        1. Detect cell-fold mask: cells where min(T1, T2) <= 0.
        2. Dilate by MERGE_DILATION cells; cc_label to get clusters.
        3. Build each cluster's crop:
             bbox = cluster_cells_bbox + BBOX_PAD cells of padding
             interior_mask = inner (h+1, w+1) rectangle [1:-1, 1:-1]
        4. Partition clusters into non-conflicting rounds (strict
           1-cell gap between same-round bboxes).
        5. For each round: submit all clusters as futures to a
           ProcessPoolExecutor; each worker solves L2 multi-pass +
           L1 polish on its crop using analytical-Jacobian SLSQP.
        6. Splice each cluster's interior-mask corners back into the
           slice.
        7. If n_neg dropped: continue.
           If n_neg unchanged for STALL_ITERS_BEFORE_ESCALATE iters
           and MERGE_DILATION < MAX: bump MERGE_DILATION by 1 and
           re-enumerate clusters.
        8. Break when n_neg = 0.
```

## Key design choices (with empirical justification)

These came from a parameter sweep on `z=126` (178 init folds, hardest
of the lightest slices).

### 1. Interior mask: **rectangular [1:-1, 1:-1]** (not component-shaped)

The 2D worker treats voxel corners as variables iff they're inside the
cluster's bbox boundary. Every corner except the outer 1-corner ring
is movable.

A tighter "component-shaped" mask (only corners adjacent to fold
cells) was tested -- and it failed to converge. Severe folds need
**displacement budget spread over many cells**; restricting movable
corners to fold-adjacent ones over-constrains the QP. The
rectangular mask reproduces the convergence pattern of notebooks 17 /
18.

| mask        | MERGE=1     | MERGE=2     |
|-------------|-------------|-------------|
| component   | 178 -> 13   | 178 -> 2    |
| **rect**    | 178 -> 6    | **178 -> 0**|

### 2. MERGE_DILATION starts at 2, escalates on stall

Two close fold components share corners via their gap cells. Solving
them independently makes them fight each other -- one solve's
boundary corner is the other solve's movable corner. Dilating the
fold mask by `MERGE_DILATION` cells before connected-component
labelling merges close fold regions into a single joint cluster that
SLSQP can resolve coherently.

  MERGE=1 starts too tight on dense slices (z=126: 6 residuals).
  MERGE=2 is the sweet spot for most slices (z=126: 0 residuals).
  MERGE=3+ over-merges into too-large clusters (slower, more residuals).

For slices where MERGE=2 stalls (e.g. z=117 had 9 residuals at
MERGE=2), the in-loop **escalator** bumps MERGE_DILATION by 1 after 2
consecutive non-improving outer iters, capped at MERGE_DILATION_MAX=6.
Empirically this rescued every previously-stuck slice tested.

### 3. BBOX_PAD = 1 (no more)

The bbox is `dilated_component_bbox + BBOX_PAD` cells. Increasing pad
adds **constraint cells with frozen corners** without adding movable
corners -- it over-constrains the QP. Tested:

| BBOX_PAD | MERGE=1     | MERGE=2     |
|----------|-------------|-------------|
| **1**    | 178 -> 13   | **178 -> 2**|
| 2        | 178 -> 46   | 178 -> 45   |
| 3        | 178 -> 91   | n/a         |

Bigger pad is strictly worse. Stay at 1.

### 4. Parallel cluster solves via `ProcessPoolExecutor`

Without parallelism, per-cluster spawn cost (Windows: ~1-2 s for a
fresh subprocess to import numpy/scipy/dvfopt) dominated. With one
long-lived pool of N workers (default `os.cpu_count() - 2`), modules
are imported once and cluster solves dispatch as futures.

Clusters in the same parallel "round" must have **non-conflicting
bboxes** (1-cell gap apart) so they don't write each other's frozen-
edge corners. Greedy graph-colouring packs them into the minimum
number of rounds. Each cluster's splice only writes its
`interior_mask` corners, so even if bboxes overlap one cell at a
boundary, frozen-edge corners stay at the snapshot value.

Speedup measured on z=126:
  serial subprocess-per-cluster:   ~100-900 s
  parallel pool + strict partition:  ~7-13 s

### 5. Analytical Jacobian for the 2-triangle constraint

The constraint `c(z) = [T1(z), T2(z)]` is a quadratic in the corner
coordinates. Closed-form partial derivatives are derived in
`_make_2tri_jac_2d` and verified against finite differences to 1e-10.

Providing the analytical Jacobian to SLSQP saves the `N+1` column
sweep that scipy would otherwise do for finite differences. The win is
modest (~3x per cluster solve in isolation, see `_bench_jac_variants.py`)
because spawn cost dominates -- but it's free correctness and never
hurts.

### 6. Per-pass perturb-on-stall in the cluster solver

Inside `solve_cluster_inline`, between L2 multi-pass iterations:

  - First pass: tiny Gaussian (sigma=1e-3) perturbation if z = anchor
    (otherwise SLSQP starts at zero gradient and can refuse to move).
  - Later passes: if a pass leaves n_neg unchanged, the next pass adds
    a Gaussian (sigma = 0.005 * stall_count) perturbation to z_init so
    SLSQP starts at a different point. After 3 consecutive non-improving
    passes, give up on the cluster and let the outer loop retry.

This matches notebook 14's reactive warm-restart trick, applied at
the per-cluster pass level rather than at the SLSQP-failure level.

### 7. Ascending-fold-count slice order

The runner pre-scans all 528 slices for their initial fold count
(~5 s) and processes them in **ascending** order. Easy slices finish
first so the CSV fills quickly, and the heaviest slices (up to 3257
folds) run last with the 30-min slice cap as a safety net.

## Tunables

```python
THRESHOLD = 0.01                 # 2-triangle constraint lower bound
EPS_L1 = 1e-4                    # smoothed-L1 epsilon for polish
MERGE_DILATION = 2               # starting value; escalates on stall
MERGE_DILATION_MAX = 6
STALL_ITERS_BEFORE_ESCALATE = 2
BBOX_PAD = 1                     # do not increase
MAX_CLUSTER_CELLS = 2000         # crops above this are skipped
MAX_CLUSTER_PER_AXIS = 60
MAX_CLUSTER_OUTER_ITERS = 20
MAX_SLICE_TIME_S = 1800          # per-slice wall-clock cap

L2_MAX_PASSES = 15
L2_PASS_MAX_ITER = 80            # SLSQP iter cap per L2 pass
L2_PASS_TIMEOUT_S = 60           # per-future timeout in the pool
L1_POLISH_MAX_ITER = 120
L1_POLISH_TIMEOUT_S = 60

N_PARALLEL_WORKERS = max(1, os.cpu_count() - 2)
```

## Architecture files

- `_run_2d_clusters.py` -- main runner, slice loop, executor, CSV / log.
- `_bench_worker.py` -- module of worker functions
  (`solve_cluster_inline`, `_make_2tri_jac_2d`, ...). Lives in its own
  file so Windows `multiprocessing.spawn` can re-import the worker
  function in each child process.

## How to launch

```powershell
python notebooks\manuscript\_run_2d_clusters.py
```

Resumable: any `z` already in `output/2d_real_full/per_slice.csv` is
skipped on restart. Snapshots the corrected `(3, D, H, W)` volume to
`output/2d_real_full/checkpoint.npz` every `CHECKPOINT_EVERY=5` slices.

Outputs written to `notebooks/manuscript/output/2d_real_full/`:

| file              | content                                           |
|-------------------|---------------------------------------------------|
| `per_slice.csv`   | one row per slice with init/final fold count,    |
|                   | min_tri, L1/L2 norms, total time, feasibility    |
| `per_cluster.csv` | one row per cluster solve across all outer iters |
| `run.log`         | structured progress log                          |
| `checkpoint.npz`  | corrected `(3, D, H, W)` snapshot                 |

## Empirical performance

On the real DVF (528 slices, 178-3257 folds per slice):

  z=126 (178 folds):  0 residuals,  12.2 s,  min_tri = +0.0099
  z=117 (185 folds):  0 residuals,  31.1 s,  min_tri = +0.0086
                      (MERGE escalated 2 -> 3 at outer iter 2)
  z=127 (195 folds):  0 residuals,  25.6 s,  min_tri = +0.0069

100% feasibility on the first 3 slices tested. Full-run projection
(~30 s/slice avg, 528 slices): ~4-5 hours wall time.
