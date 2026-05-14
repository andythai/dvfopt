# Solver Benchmarks

Direct comparisons of the `dvfopt` correction solvers on shared test
fields. These notebooks do **not** involve external registration
algorithms — they stress only the optimiser.

| Subfolder | Solver family |
|-----------|---------------|
| [slsqp/](slsqp/) | SciPy SLSQP (windowed iterative + full-grid), 2D and 3D. |
| [barrier/](barrier/) | Penalty → log-barrier L-BFGS-B (`iterative_3d_barrier`), CPU and GPU. |

See each subfolder's README for per-notebook descriptions.

## Common structure

Every notebook in this tree:

1. Inserts `benchmarks/` onto `sys.path` so it can import
   [`../benchmark_utils`](../benchmark_utils.py).
2. Calls `get_output_dir(METHOD, NOTEBOOK_NAME, base="../../output")`.
3. Calls `log_run_header(...)` / `log_run_footer(...)` around the run.
4. Saves every figure via `show_and_save(OUTPUT_DIR)` and writes a
   `results.csv` plus `summary.json` at the end.
