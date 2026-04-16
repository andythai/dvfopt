# Benchmarks

Performance and correctness benchmarks for the `dvfopt` correction methods
and for external registration algorithms paired with post-hoc correction.

Each notebook is self-contained and writes a per-run output folder under
`output/<method>/<notebook_name>/` containing figures (PNG), a `results.csv`
table, and a `summary.json` run manifest.

## Layout

| Folder | Purpose |
|--------|---------|
| [solvers/slsqp/](solvers/slsqp/) | SLSQP-based correction: serial vs parallel, constraint modes, windowed vs full-grid, 3D correction. |
| [solvers/barrier/](solvers/barrier/) | Penalty → log-barrier L-BFGS solver: 3D, windowed vs full-grid, CPU vs GPU. |
| [scaling/](scaling/) | Performance vs grid size, folding severity, and L2 ↔ Jdet correlation. Both SLSQP and barrier variants. |
| [registration/](registration/) | External registration algorithms (Elastix, ANTs, VoxelMorph, TransMorph, OpenCV) + post-hoc correction. |
| [pipelines/](pipelines/) | End-to-end 3D slice-wise correction pipelines (serial, parallel, multi-constraint). |

## Shared utilities

### [benchmark_utils.py](benchmark_utils.py)

Helpers imported by every notebook. Notebooks add `..` (or deeper) to
`sys.path` so they can import this module from any subfolder.

**Output management**

- `get_output_dir(method, notebook_name, base="output")` — create and return
  `output/<method>/<notebook_name>/`.
- `save_figure(fig, output_dir, name, dpi=150, close=False)` — persist a
  matplotlib figure as PNG.
- `save_results_csv(rows, columns, output_dir, name="results")` — write
  tabular results as CSV.
- `save_summary_json(data, output_dir, name="summary")` — write a JSON
  run manifest.
- `show_and_save(output_dir, name=None, fig=None, dpi=150)` — drop-in
  replacement for `plt.show()` that also persists the figure
  (auto-increments `figure_01`, `figure_02`, …).
- `reset_figure_counter()` — reset the auto-increment counter used by
  `show_and_save`.
- `log_run_header(method, notebook_name, output_dir, extra=None)` /
  `log_run_footer(summary, results)` — print a standardised banner and
  return a summary dict suitable for `save_summary_json`.
- `results_to_rows(results, extra_cols=None)` — flatten a result dict
  `{label: {metric: value}}` into `(rows, columns)` for CSV export.

**Metric collection**

- `run_correction(dvf, solver, verbose=0, **kwargs)` — run a 2D solver
  against a `(3, 1, H, W)` DVF and return the standard metric bundle
  (`phi_init`, `phi`, `jac_init`, `jac_final`, `time`, `n_neg_init`,
  `n_neg_final`, `min_jdet_init`, `min_jdet`, `l2_err`).
- `run_correction_3d(dvf, solver, verbose=0, **kwargs)` — 3D variant for
  `(3, D, H, W)` fields.

**Plotting**

- `plot_jac_heatmaps(jac_grid, col_labels, row_labels, title, figscale)`
  — grid of Jdet heatmaps with a shared diverging colormap and zero-level
  contour.
- `plot_correction_magnitude(phi_pairs, labels, title, figscale)` —
  per-pixel correction-magnitude heatmaps `|φ − φ_init|`.
- `plot_jdet_histograms(jac_groups, labels, title, figscale, colors)` —
  overlaid Jdet distribution histograms.

### [_full_volume_parallel.py](_full_volume_parallel.py)

Worker-side helpers for parallel full-volume 3D correction. Module-level
functions are required for `ProcessPoolExecutor` on Windows (spawn start
method serialises the target callable).

- `solve_group(args)` — sequentially correct a list of patches within a
  single region. Returns `(region_bbox, phi_region, pre_group_neg,
  post_group_neg)`.

## Output convention

```
output/
  <method>/                # e.g. "slsqp", "barrier", "barrier-gpu", "registration"
    <notebook_name>/       # e.g. "scalability", "3d-correction"
      figure_01.png
      figure_02.png
      ...
      results.csv
      summary.json
```

All benchmark notebooks call `reset_figure_counter()` at the top so
`figure_NN.png` filenames are stable across re-runs.

## Dependencies

Most benchmarks run on the core `dvfopt` install. A few require extras:

```bash
pip install -e ".[benchmarks]"  # itk-elastix, opencv, timm, torch, voxelmorph
```

See the per-folder READMEs for which extras each notebook needs.
