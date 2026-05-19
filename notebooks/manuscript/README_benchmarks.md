# Manuscript benchmark notebooks: 2D per-slice + 3D full-volume

Two L2-then-L1 benchmarks on the real registration field at
`data/corrected_correspondences_count_touching/registered_output/deformation3d.npy`.

## What runs

- **`03_benchmark_2d_real_full.ipynb`** — runs the 2-triangle constraint
  with windowed SLSQP + L1 polish on every z-slice (528 total). Same
  pipeline available as a stand-alone Python script (`_run_2d_full.py`)
  so it can run unattended.
- **`04_benchmark_3d_real_full.ipynb`** — runs the 6-tet constraint with
  windowed SLSQP + L1 polish on the full volume. *Current status:* the
  windowed solver is not converging on this field (windows are too
  small relative to the fold structure). Needs window-strategy tuning
  before launching the full run.

Both notebooks (and the stand-alone script) save results incrementally
to `notebooks/manuscript/output/{2d_real_full,3d_real_full}/`:

- `per_slice.csv` (2D) / `trajectory.csv` (3D)
- `checkpoint.npz` — current corrected `(3, D, H, W)` field
- `run.log` — per-slice progress log
- `summary.json` plus PDF/PNG summary plots after the run

## How to launch

Open separate terminals in this repo and run:

```powershell
# 2D benchmark (per-slice, 528 slices)
python notebooks\manuscript\_run_2d_full.py

# 3D benchmark (fold-clustered, processes one crop per connected fold component)
python notebooks\manuscript\_run_3d_clusters.py
```

`_run_3d_clusters.py` is the recommended 3D entrypoint. It pre-computes
the connected components of folded cells in the volume, then runs
notebook-18-style multi-pass full-grid L2 + L1 polish on each
component's bounding-box crop. Components sorted by size desc, so a
partial run still hits the heaviest fold regions first. Components
whose bbox exceeds the SLSQP-tractable size are flagged ``skipped`` in
the CSV (still informative -- shows where the recipe runs out of
headroom).

The earlier ``_run_3d_full.py`` (windowed) and ``_run_3d_crops.py``
(per-cell full-scan) entrypoints are kept for reference but not
recommended for the full real DVF -- the windowed solver hits the
boundary-freeze degeneracy, and the per-cell full-scan generates
~350K crops which is impractical even with fast skip paths.

That's it. The script writes `output/2d_real_full/run.log` and the CSV
as it goes. It is **resumable**: any z's already in `per_slice.csv` are
skipped on re-launch, so a kill-and-restart loses at most the slice
currently in progress plus the corrected-volume state for the slices
between the last `checkpoint.npz` save (every 10 slices) and the kill.

Approximate runtime per slice based on smoke tests:
- "easy" slices (~few hundred folded triangles): 30 - 90 s with full clear.
- "hard" slices (1000 - 3000+ folded triangles): 1 - 3 min, often
  partial -- the runner moves on rather than waste hours on one slice.

Total 528-slice run estimate: 12 - 36 hours depending on fold-count
distribution.

To monitor progress while it runs:

```powershell
Get-Content -Wait notebooks\manuscript\output\2d_real_full\run.log
```

## Pipeline details (both notebooks)

Per slice (2D) / per window (3D):

1. **L2 phase** -- pick worst-fold pixel/voxel, build connected-component
   bbox capped at `MAX_WINDOW_PER_AXIS` cells per axis, run L2-objective
   SLSQP locally with the 2-triangle (2D) or 6-tet (3D) constraint.
   Frozen-edge boundary; interior pixels are the SLSQP variables.
2. **Perturb-on-stall** -- if the outer loop stalls for
   `L2_STALL_LIMIT` consecutive iters without progress, apply a Gaussian
   sigma=0.02 perturbation to the folded neighbourhood and retry. Same
   idea as notebook 14's reactive warm-restart, hoisted to slice level.
3. **L1 polish phase** -- after L2 reaches feasibility, find connected
   components of touched cells; for each, re-solve the sub-window with
   the smoothed-L1 objective. Accept the polish only if it strictly
   reduces the L1 norm and doesn't re-introduce folds.

Every SLSQP call runs in a **child process with a wall-clock timeout**
(30 s by default). On timeout, the child is killed and the outer loop
moves on. This is necessary because scipy's SLSQP can hang at the
Fortran active-set QP level on pathological windows; `max_minimize_iter`
only bounds the outer SQP loop, not the inner QP. The full 30-min hang
of the very first smoke run was an example of this.

## Known caveats

- **3D windowed solver not converging on full volume.** The 6-tet
  constraint is harder to satisfy locally than the 2-triangle one --
  many folds in this field span multiple "windows", and frozen-boundary
  cells leave SLSQP without enough degrees of freedom to fix the
  near-boundary tets. The 2D notebook works fine because the 2-triangle
  constraint at a cell only involves 4 corners, so smaller windows
  suffice.
- **OOM if window cap is removed.** Earlier smoke runs that called
  `iterative_serial` directly (no external window cap) tried to
  allocate 270 GiB on slice z=2 (3257 folded triangles). The current
  pipeline caps windows at 14x14 cells (= ~400 variables, well under
  SLSQP's practical limit) which avoids the OOM.

## Files in this folder

| File | Purpose |
|------|---------|
| `01_signed_areas.ipynb` | manuscript figure |
| `02_tet_decomposition.ipynb` | manuscript figure |
| `03_benchmark_2d_real_full.ipynb` | 2D per-slice benchmark notebook |
| `04_benchmark_3d_real_full.ipynb` | 3D full-volume benchmark notebook |
| `_bench_worker.py` | picklable SLSQP worker functions |
| `_run_2d_full.py` | stand-alone 2D runner |
| `_smoke_2d.py` | 2D smoke test (3 representative slices) |
| `_smoke_3d.py` | 3D smoke test on small ROI |
