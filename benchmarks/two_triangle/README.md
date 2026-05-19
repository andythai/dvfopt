# Two-Triangle Benchmark Harness

Standalone benchmarking for candidate optimizations of the 2-triangle / 2-tetrahedra
fold-correction pipeline. Lives outside `dvfopt/` so experiments can iterate freely.

## Run a sweep

```bash
# Smoke test
python -m benchmarks.two_triangle.runner --variants baseline_serial --cases synth2d_single_cell_flip

# Full sweep
python -m benchmarks.two_triangle.runner --all
```

Results land in `benchmarks/two_triangle/results/runs/<utc-stamp>/`.
Open `report.ipynb` and re-run to refresh plots against the latest run.

## Add a variant

Create `variants/<name>.py`, decorate the entry function with `@register_variant("name")`,
and add an import line in `variants/__init__.py`.

## Add a case

Create or edit a file in `cases/`, decorate with `@register_case("name", category=...)`,
and add an import line in `cases/__init__.py`.

## Tests

```bash
pytest benchmarks/two_triangle/tests -v
```
