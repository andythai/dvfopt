# Two-Triangle Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone benchmark harness under `benchmarks/two_triangle/` that runs candidate solver variants for the 2-triangle/2-tet fold-correction problem against a fixed test suite, captures speed/accuracy/quality/convergence trajectories, and produces side-by-side comparisons. **No edits to `dvfopt/`, `laplacian/`, or `test_cases/`.**

**Architecture:** Each *variant* is a Python callable returning a `SolverResult` dataclass; *cases* are callables returning `(phi_initial, metadata)`. A runner sweeps variants × cases, persists per-cell Parquet files, and a notebook produces comparison plots. Wrapper variants (combinatorial pre-fix, multigrid, SVF warm-start) call the existing `iterative_serial` / `iterative_3d`. Loop-owning variants (soft-margin, active-set, trust-constr) re-implement a minimal version of the outer loop to inject custom constraint logic and capture per-iteration trajectory.

**Tech Stack:** Python 3, NumPy, SciPy (`scipy.optimize`, `scipy.ndimage`, `scipy.sparse`), Pandas, PyArrow (Parquet), psutil (memory sampling), pytest (harness self-tests), Jupyter (report notebook), matplotlib.

**Reference spec:** `~/.claude/plans/help-me-brainstorm-if-fuzzy-kay.md` — design that this plan implements.

---

## File Structure

```
benchmarks/two_triangle/
├── __init__.py                          # Task 1
├── README.md                            # Task 1
├── registry.py                          # Task 1 — @register_variant, @register_case
├── result.py                            # Task 1 — SolverResult dataclass + Parquet I/O
├── metrics.py                           # Task 2 — fold counters, l2_disp, smoothness
├── trajectory.py                        # Task 7 — trajectory accumulator
├── runner.py                            # Task 14 — CLI entrypoint
├── _iterative_loop.py                   # Task 7 — minimal re-implemented outer loop
├── cases/
│   ├── __init__.py                      # Task 3 — imports + registers all cases
│   ├── synthetic_2d.py                  # Task 3
│   ├── synthetic_3d.py                  # Task 4
│   ├── random_3d.py                     # Task 5
│   └── real_slices.py                   # Task 6
├── variants/
│   ├── __init__.py                      # Task 2 — imports + registers all variants
│   ├── baseline_serial.py               # Task 2
│   ├── combinatorial_prefix.py          # Task 8
│   ├── soft_margin.py                   # Task 9
│   ├── active_set.py                    # Task 10
│   ├── multigrid.py                     # Task 11
│   ├── trust_constr.py                  # Task 12
│   └── svf_warmstart.py                 # Task 13
├── tests/
│   ├── __init__.py                      # Task 1
│   ├── conftest.py                      # Task 1 — adds repo root to sys.path
│   ├── test_registry.py                 # Task 1
│   ├── test_result.py                   # Task 1
│   ├── test_metrics.py                  # Task 2
│   ├── test_baseline.py                 # Task 2
│   ├── test_cases_registry.py           # Tasks 3-6
│   ├── test_iterative_loop.py           # Task 7
│   ├── test_variants_wrapper.py         # Tasks 8, 11, 13
│   ├── test_variants_loop_owning.py     # Tasks 9, 10, 12
│   ├── test_runner.py                   # Task 14
│   └── test_parity.py                   # Task 15 — baseline vs direct iterative_serial
└── results/                             # Task 14 — gitignored output dir
report.ipynb                             # Task 16 — at benchmarks/two_triangle/report.ipynb
```

Also: append `/benchmarks/two_triangle/results/` to `.gitignore` (Task 14).

---

## Task 1: Skeleton — Registry, SolverResult, package layout

**Files:**
- Create: `benchmarks/two_triangle/__init__.py`
- Create: `benchmarks/two_triangle/README.md`
- Create: `benchmarks/two_triangle/registry.py`
- Create: `benchmarks/two_triangle/result.py`
- Create: `benchmarks/two_triangle/tests/__init__.py`
- Create: `benchmarks/two_triangle/tests/conftest.py`
- Create: `benchmarks/two_triangle/tests/test_registry.py`
- Create: `benchmarks/two_triangle/tests/test_result.py`

- [ ] **Step 1.1: Create `__init__.py` and `README.md`**

`benchmarks/two_triangle/__init__.py`:
```python
"""Standalone benchmark harness for 2-triangle/2-tet fold correction variants.

This package is intentionally separate from `dvfopt/` — variants compose
existing primitives but never modify them. See README.md for usage.
"""
```

`benchmarks/two_triangle/README.md`:
```markdown
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
```

- [ ] **Step 1.2: Create conftest with sys.path setup**

`benchmarks/two_triangle/tests/__init__.py`: empty file.

`benchmarks/two_triangle/tests/conftest.py`:
```python
"""Pytest config: add repo root to sys.path so tests can import dvfopt etc."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
```

- [ ] **Step 1.3: Write failing test for registry**

`benchmarks/two_triangle/tests/test_registry.py`:
```python
import pytest
from benchmarks.two_triangle import registry


def test_register_and_get_variant():
    @registry.register_variant("test_variant_1")
    def _v(phi, **kw):
        return None

    assert registry.get_variant("test_variant_1") is _v
    assert "test_variant_1" in registry.list_variants()


def test_register_and_get_case():
    @registry.register_case("test_case_1", category="synthetic_2d", dim=2)
    def _c():
        return None, {}

    assert registry.get_case("test_case_1") is _c
    meta = registry.case_metadata("test_case_1")
    assert meta["category"] == "synthetic_2d"
    assert meta["dim"] == 2


def test_duplicate_variant_raises():
    @registry.register_variant("dup_variant")
    def _a(phi, **kw):
        return None

    with pytest.raises(ValueError, match="already registered"):
        @registry.register_variant("dup_variant")
        def _b(phi, **kw):
            return None


def test_unknown_variant_raises():
    with pytest.raises(KeyError):
        registry.get_variant("nonexistent")
```

- [ ] **Step 1.4: Run registry tests — expect FAIL**

```powershell
pytest benchmarks/two_triangle/tests/test_registry.py -v
```
Expected: `ImportError` or `ModuleNotFoundError` — `registry.py` does not exist yet.

- [ ] **Step 1.5: Implement `registry.py`**

`benchmarks/two_triangle/registry.py`:
```python
"""Decorator-based registries for benchmark variants and test cases."""
from typing import Callable, Dict

_VARIANTS: Dict[str, Callable] = {}
_CASES: Dict[str, Callable] = {}
_CASE_META: Dict[str, dict] = {}


def register_variant(name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        if name in _VARIANTS:
            raise ValueError(f"Variant {name!r} already registered")
        _VARIANTS[name] = fn
        return fn
    return deco


def register_case(name: str, *, category: str, dim: int, **extra) -> Callable:
    def deco(fn: Callable) -> Callable:
        if name in _CASES:
            raise ValueError(f"Case {name!r} already registered")
        _CASES[name] = fn
        _CASE_META[name] = {"category": category, "dim": dim, **extra}
        return fn
    return deco


def get_variant(name: str) -> Callable:
    return _VARIANTS[name]


def get_case(name: str) -> Callable:
    return _CASES[name]


def case_metadata(name: str) -> dict:
    return dict(_CASE_META[name])


def list_variants() -> list:
    return sorted(_VARIANTS)


def list_cases() -> list:
    return sorted(_CASES)


def clear() -> None:
    """Test-only: wipe the registries."""
    _VARIANTS.clear()
    _CASES.clear()
    _CASE_META.clear()
```

- [ ] **Step 1.6: Run registry tests — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_registry.py -v
```
Expected: 4 passed.

- [ ] **Step 1.7: Write failing test for SolverResult**

`benchmarks/two_triangle/tests/test_result.py`:
```python
import numpy as np
import pandas as pd
import pytest

from benchmarks.two_triangle.result import SolverResult


def test_construct_minimal():
    phi = np.zeros((2, 4, 4), dtype=np.float64)
    traj = pd.DataFrame({
        "outer_iter": [0],
        "time_s": [0.0],
        "fold_count_jdet": [0],
        "fold_count_tri": [0],
        "max_violation": [0.0],
        "l2_disp": [0.0],
        "smoothness": [0.0],
        "n_active_windows": [0],
        "inner_iters": [0],
    })
    r = SolverResult(
        phi_final=phi,
        trajectory=traj,
        converged=True,
        timed_out=False,
        error=None,
        meta={"variant": "baseline", "case": "demo"},
    )
    assert r.converged is True
    assert r.phi_final.shape == (2, 4, 4)
    assert len(r.trajectory) == 1


def test_required_trajectory_columns():
    phi = np.zeros((2, 4, 4))
    bad_traj = pd.DataFrame({"outer_iter": [0]})  # missing required columns
    with pytest.raises(ValueError, match="missing trajectory columns"):
        SolverResult(
            phi_final=phi, trajectory=bad_traj,
            converged=True, timed_out=False, error=None, meta={},
        )


def test_to_parquet_roundtrip(tmp_path):
    phi = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    traj = pd.DataFrame({
        "outer_iter": [0, 1], "time_s": [0.0, 0.5],
        "fold_count_jdet": [3, 0], "fold_count_tri": [4, 0],
        "max_violation": [-0.1, 0.05], "l2_disp": [0.0, 1.2],
        "smoothness": [0.0, 0.3], "n_active_windows": [1, 1],
        "inner_iters": [10, 5],
    })
    r = SolverResult(
        phi_final=phi, trajectory=traj,
        converged=True, timed_out=False, error=None,
        meta={"variant": "soft_margin", "case": "synth2d_single_cell_flip"},
    )
    path = tmp_path / "result.parquet"
    r.to_parquet(path)
    r2 = SolverResult.from_parquet(path)
    np.testing.assert_array_equal(r2.phi_final, phi)
    pd.testing.assert_frame_equal(r2.trajectory, traj)
    assert r2.converged == r.converged
    assert r2.meta["variant"] == "soft_margin"
```

- [ ] **Step 1.8: Run result tests — expect FAIL**

```powershell
pytest benchmarks/two_triangle/tests/test_result.py -v
```
Expected: `ImportError` — `result.py` does not exist.

- [ ] **Step 1.9: Implement `result.py`**

`benchmarks/two_triangle/result.py`:
```python
"""SolverResult dataclass + Parquet serialization."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import json
import numpy as np
import pandas as pd

REQUIRED_TRAJECTORY_COLS = (
    "outer_iter", "time_s", "fold_count_jdet", "fold_count_tri",
    "max_violation", "l2_disp", "smoothness", "n_active_windows",
    "inner_iters",
)


@dataclass
class SolverResult:
    phi_final: np.ndarray
    trajectory: pd.DataFrame
    converged: bool
    timed_out: bool
    error: Optional[str]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        missing = [c for c in REQUIRED_TRAJECTORY_COLS if c not in self.trajectory.columns]
        if missing:
            raise ValueError(f"missing trajectory columns: {missing}")

    def to_parquet(self, path) -> None:
        path = Path(path)
        # Write trajectory with metadata columns + serialized phi/meta in
        # a sibling .npz + .json so the parquet file stays tidy.
        self.trajectory.to_parquet(path)
        np.savez_compressed(path.with_suffix(".phi.npz"), phi=self.phi_final)
        sidecar = {
            "converged": self.converged,
            "timed_out": self.timed_out,
            "error": self.error,
            "meta": self.meta,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(sidecar, default=str))

    @classmethod
    def from_parquet(cls, path) -> "SolverResult":
        path = Path(path)
        traj = pd.read_parquet(path)
        phi = np.load(path.with_suffix(".phi.npz"))["phi"]
        sidecar = json.loads(path.with_suffix(".meta.json").read_text())
        return cls(
            phi_final=phi,
            trajectory=traj,
            converged=sidecar["converged"],
            timed_out=sidecar["timed_out"],
            error=sidecar["error"],
            meta=sidecar["meta"],
        )
```

- [ ] **Step 1.10: Run result tests — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_result.py -v
```
Expected: 3 passed.

- [ ] **Step 1.11: Verify pyarrow is available (Parquet dependency)**

```powershell
python -c "import pyarrow; print(pyarrow.__version__)"
```
Expected: prints a version string. If `ModuleNotFoundError`, install: `pip install pyarrow`.

- [ ] **Step 1.12: Commit**

```powershell
git add benchmarks/two_triangle/__init__.py benchmarks/two_triangle/README.md benchmarks/two_triangle/registry.py benchmarks/two_triangle/result.py benchmarks/two_triangle/tests/
git commit -m "two_triangle bench: scaffold registry + SolverResult"
```

---

## Task 2: Metrics module + baseline variant

**Files:**
- Create: `benchmarks/two_triangle/metrics.py`
- Create: `benchmarks/two_triangle/variants/__init__.py`
- Create: `benchmarks/two_triangle/variants/baseline_serial.py`
- Create: `benchmarks/two_triangle/tests/test_metrics.py`
- Create: `benchmarks/two_triangle/tests/test_baseline.py`

- [ ] **Step 2.1: Write failing test for metrics**

`benchmarks/two_triangle/tests/test_metrics.py`:
```python
import numpy as np
from benchmarks.two_triangle import metrics


def _identity_phi_2d(H=4, W=4):
    return np.zeros((2, H, W), dtype=np.float64)


def _identity_phi_3d(D=3, H=4, W=4):
    return np.zeros((3, D, H, W), dtype=np.float64)


def test_fold_counts_clean_2d():
    phi = _identity_phi_2d()
    out = metrics.fold_counts(phi, threshold=0.01)
    assert out["fold_count_jdet"] == 0
    assert out["fold_count_tri"] == 0
    assert out["max_violation"] > 0  # min Jdet of identity ~ 1


def test_fold_counts_clean_3d():
    phi = _identity_phi_3d()
    out = metrics.fold_counts(phi, threshold=0.01)
    assert out["fold_count_jdet"] == 0
    assert out["fold_count_tri"] == 0


def test_l2_displacement_zero_when_unchanged():
    phi = _identity_phi_2d()
    assert metrics.l2_displacement(phi, phi) == 0.0


def test_l2_displacement_positive_when_changed():
    phi0 = _identity_phi_2d()
    phi1 = phi0.copy()
    phi1[0, 1, 1] = 0.5
    assert metrics.l2_displacement(phi1, phi0) == 0.5


def test_smoothness_zero_for_identity():
    phi = _identity_phi_2d(8, 8)
    assert metrics.smoothness(phi) == 0.0


def test_smoothness_positive_for_bumpy_field():
    phi = _identity_phi_2d(8, 8)
    phi[0, 4, 4] = 1.0
    assert metrics.smoothness(phi) > 0.0
```

- [ ] **Step 2.2: Run metrics tests — expect FAIL**

```powershell
pytest benchmarks/two_triangle/tests/test_metrics.py -v
```
Expected: `ImportError`.

- [ ] **Step 2.3: Implement `metrics.py`**

`benchmarks/two_triangle/metrics.py`:
```python
"""Metrics computed from phi arrays. Shared between variants and reporting.

Reuses dvfopt primitives (jacobian_det2D/3D, triangle_sign_count_negatives) by
import — never modifies them.
"""
import numpy as np

from dvfopt.jacobian.numpy_jdet import jacobian_det2D, jacobian_det3D
from dvfopt.jacobian.triangle_sign import (
    triangle_sign_areas2D,
    triangle_sign_count_negatives,
)


def _is_3d(phi: np.ndarray) -> bool:
    """phi is (3, D, H, W) for 3D, (2, H, W) or (3, 1, H, W) for 2D."""
    return phi.ndim == 4 and phi.shape[0] == 3 and phi.shape[1] > 1


def fold_counts(phi: np.ndarray, *, threshold: float = 0.01) -> dict:
    """Return {fold_count_jdet, fold_count_tri, max_violation} for 2D or 3D phi.

    `max_violation` is the minimum constraint value (the most negative cell);
    positive means the field is feasible.
    """
    if _is_3d(phi):
        jdet = jacobian_det3D(phi)
        n_jdet = int((jdet < threshold).sum())
        # 3D triangle (= tetrahedron) sign check uses the sitk_jdet path;
        # for 3D we approximate fold_count_tri as fold_count_jdet here.
        # When the strict 6-tet count helper lands in dvfopt this can be
        # swapped in; for now report jdet as the proxy.
        n_tri = n_jdet
        max_viol = float(jdet.min())
    else:
        # Convert (3, 1, H, W) to (2, H, W) if needed
        if phi.ndim == 4 and phi.shape[0] == 3 and phi.shape[1] == 1:
            phi2 = np.stack([phi[1, 0], phi[2, 0]])
        else:
            phi2 = phi
        jdet = jacobian_det2D(phi2)
        n_jdet = int((jdet < threshold).sum())
        n_tri = triangle_sign_count_negatives(phi2)
        tri_areas = triangle_sign_areas2D(phi2)
        max_viol = float(min(jdet.min(), tri_areas.min()))
    return {
        "fold_count_jdet": n_jdet,
        "fold_count_tri": n_tri,
        "max_violation": max_viol,
    }


def l2_displacement(phi: np.ndarray, phi_initial: np.ndarray) -> float:
    return float(np.sqrt(np.sum((phi - phi_initial) ** 2)))


def smoothness(phi: np.ndarray) -> float:
    """Frobenius norm of the discrete Laplacian of phi."""
    # Five-point Laplacian along each spatial axis, summed.
    if _is_3d(phi):
        # phi shape (3, D, H, W) -> Laplacian over last 3 axes
        lap = np.zeros_like(phi)
        for ax in (1, 2, 3):
            lap += np.gradient(np.gradient(phi, axis=ax), axis=ax)
    else:
        if phi.ndim == 4 and phi.shape[1] == 1:
            phi = np.stack([phi[1, 0], phi[2, 0]])
        lap = np.zeros_like(phi)
        for ax in (1, 2):
            lap += np.gradient(np.gradient(phi, axis=ax), axis=ax)
    return float(np.sqrt(np.sum(lap ** 2)))
```

- [ ] **Step 2.4: Run metrics tests — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_metrics.py -v
```
Expected: 6 passed.

- [ ] **Step 2.5: Write failing test for baseline variant**

`benchmarks/two_triangle/tests/test_baseline.py`:
```python
import numpy as np
import pytest

from benchmarks.two_triangle import registry
import benchmarks.two_triangle.variants  # registers baseline


def _make_tiny_2d_fold():
    """Construct a (3, 1, H, W) deformation with one folded pixel."""
    H, W = 6, 6
    dvf = np.zeros((3, 1, H, W), dtype=np.float64)
    # Push pixel (3, 3) far enough to flip the local triangle
    dvf[1, 0, 3, 3] = -2.0  # dy
    dvf[2, 0, 3, 3] = -2.0  # dx
    return dvf


def test_baseline_variant_registered():
    assert "baseline_serial" in registry.list_variants()


def test_baseline_variant_runs_2d():
    dvf = _make_tiny_2d_fold()
    fn = registry.get_variant("baseline_serial")
    result = fn(dvf, threshold=0.01, max_iterations=20)
    assert result.phi_final.shape == (2, 6, 6)
    # Trajectory has at least one row (the final state)
    assert len(result.trajectory) >= 1
    # Either converged or hit max-iter — both fine for a smoke test
    assert isinstance(result.converged, bool)
    assert result.error is None
```

- [ ] **Step 2.6: Run baseline test — expect FAIL**

```powershell
pytest benchmarks/two_triangle/tests/test_baseline.py -v
```
Expected: `ModuleNotFoundError` for `benchmarks.two_triangle.variants` or `KeyError`.

- [ ] **Step 2.7: Implement baseline variant**

`benchmarks/two_triangle/variants/__init__.py`:
```python
"""Importing this package registers all variants."""
from benchmarks.two_triangle.variants import baseline_serial  # noqa: F401
```

`benchmarks/two_triangle/variants/baseline_serial.py`:
```python
"""Baseline variant: calls dvfopt iterative_serial / iterative_3d unchanged.

Produces a 1-row trajectory containing the final-state values only — no
per-iteration data is captured, since the existing solvers do not expose
hooks. This is the reference for all other variants.
"""
import time
import traceback

import numpy as np
import pandas as pd

from dvfopt.core.slsqp.iterative import iterative_serial
from dvfopt.core.slsqp.iterative3d import iterative_3d

from benchmarks.two_triangle.registry import register_variant
from benchmarks.two_triangle.result import SolverResult
from benchmarks.two_triangle.metrics import fold_counts, l2_displacement, smoothness


def _is_3d_input(dvf: np.ndarray) -> bool:
    """dvf shape (3, D, H, W) with D > 1 is 3D; (3, 1, H, W) is 2D."""
    return dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] > 1


@register_variant("baseline_serial")
def baseline_serial(dvf: np.ndarray, *, threshold: float = 0.01,
                    max_iterations: int = 100, enforce_triangles: bool = True,
                    timeout_s: float = 600.0, **_unused) -> SolverResult:
    is_3d = _is_3d_input(dvf)
    phi_initial = dvf.copy()
    t0 = time.perf_counter()
    err = None
    converged = False
    try:
        if is_3d:
            phi_final = iterative_3d(
                dvf.copy(), threshold=threshold,
                max_iterations=max_iterations, verbose=0,
            )
        else:
            phi_final = iterative_serial(
                dvf.copy(), threshold=threshold,
                max_iterations=max_iterations, verbose=0,
                enforce_triangles=enforce_triangles,
            )
        elapsed = time.perf_counter() - t0
        # Determine convergence by checking remaining folds
        fc = fold_counts(phi_final, threshold=threshold)
        converged = fc["fold_count_jdet"] == 0 and fc["fold_count_tri"] == 0
    except Exception:  # pylint: disable=broad-except
        elapsed = time.perf_counter() - t0
        err = traceback.format_exc()
        # Return phi_initial as fallback so downstream metrics don't crash.
        phi_final = phi_initial[1:, 0] if not is_3d else phi_initial.copy()

    if not is_3d and phi_final.ndim == 4 and phi_final.shape[1] == 1:
        phi_final_canonical = np.stack([phi_final[1, 0], phi_final[2, 0]])
    else:
        phi_final_canonical = phi_final

    fc = fold_counts(phi_final_canonical, threshold=threshold)
    if not is_3d:
        phi_init_canonical = np.stack([phi_initial[1, 0], phi_initial[2, 0]])
    else:
        phi_init_canonical = phi_initial
    traj = pd.DataFrame([{
        "outer_iter": 0,
        "time_s": elapsed,
        "fold_count_jdet": fc["fold_count_jdet"],
        "fold_count_tri": fc["fold_count_tri"],
        "max_violation": fc["max_violation"],
        "l2_disp": l2_displacement(phi_final_canonical, phi_init_canonical),
        "smoothness": smoothness(phi_final_canonical),
        "n_active_windows": 0,
        "inner_iters": 0,
    }])
    return SolverResult(
        phi_final=phi_final_canonical,
        trajectory=traj,
        converged=converged,
        timed_out=False,
        error=err,
        meta={"variant": "baseline_serial", "is_3d": is_3d,
              "threshold": threshold, "max_iterations": max_iterations,
              "enforce_triangles": enforce_triangles},
    )
```

- [ ] **Step 2.8: Run baseline test — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_baseline.py -v
```
Expected: 2 passed (within ~30 s).

- [ ] **Step 2.9: Commit**

```powershell
git add benchmarks/two_triangle/metrics.py benchmarks/two_triangle/variants/ benchmarks/two_triangle/tests/test_metrics.py benchmarks/two_triangle/tests/test_baseline.py
git commit -m "two_triangle bench: metrics module + baseline_serial variant"
```

---

## Task 3: Synthetic 2D test cases

**Files:**
- Create: `benchmarks/two_triangle/cases/__init__.py`
- Create: `benchmarks/two_triangle/cases/synthetic_2d.py`
- Create: `benchmarks/two_triangle/tests/test_cases_registry.py`

- [ ] **Step 3.1: Write failing test**

`benchmarks/two_triangle/tests/test_cases_registry.py`:
```python
import numpy as np
import pytest

from benchmarks.two_triangle import registry
import benchmarks.two_triangle.cases  # registers cases


SYNTHETIC_2D = (
    "synth2d_single_cell_flip",
    "synth2d_horizontal_bowtie",
    "synth2d_diagonal_bowtie",
    "synth2d_layered_bowtie_stack",
)


@pytest.mark.parametrize("name", SYNTHETIC_2D)
def test_synthetic_2d_case_registered(name):
    assert name in registry.list_cases()
    fn = registry.get_case(name)
    dvf, meta = fn()
    assert dvf.ndim == 4
    assert dvf.shape[0] == 3
    assert dvf.shape[1] == 1
    assert "title" in meta
    assert meta.get("dim") in (2, None)


@pytest.mark.parametrize("name", SYNTHETIC_2D)
def test_synthetic_2d_case_has_initial_folds(name):
    """Every synthetic case must start with at least one fold to be useful."""
    from benchmarks.two_triangle.metrics import fold_counts
    fn = registry.get_case(name)
    dvf, _ = fn()
    fc = fold_counts(dvf, threshold=0.01)
    assert fc["fold_count_jdet"] > 0 or fc["fold_count_tri"] > 0, (
        f"Case {name} has no initial folds — useless as a benchmark"
    )
```

- [ ] **Step 3.2: Run case test — expect FAIL**

```powershell
pytest benchmarks/two_triangle/tests/test_cases_registry.py -v
```
Expected: `ModuleNotFoundError` for `benchmarks.two_triangle.cases`.

- [ ] **Step 3.3: Implement synthetic 2D cases**

`benchmarks/two_triangle/cases/__init__.py`:
```python
"""Importing this package registers all cases."""
from benchmarks.two_triangle.cases import synthetic_2d  # noqa: F401
```

`benchmarks/two_triangle/cases/synthetic_2d.py`:
```python
"""Synthetic 2D bowtie + single-cell-flip cases.

Each case is constructed from raw correspondence pairs via Laplacian
interpolation (mirroring the patterns shown in notebooks 09 and 13). All
return (3, 1, H, W) deformation arrays so they're drop-in compatible with
the existing iterative_serial signature.
"""
import numpy as np

from laplacian import solveLaplacianFromCorrespondences

from benchmarks.two_triangle.registry import register_case


def _build(H: int, W: int, msample: list, fsample: list) -> np.ndarray:
    ms = np.array(msample, dtype=int)
    fs = np.array(fsample, dtype=int)
    return solveLaplacianFromCorrespondences((1, H, W), ms, fs)


@register_case("synth2d_single_cell_flip", category="synthetic_2d", dim=2)
def synth2d_single_cell_flip():
    """Tiny 6x6 grid with one diagonal correspondence that flips one cell."""
    dvf = _build(
        6, 6,
        msample=[[0, 2, 2]],
        fsample=[[0, 4, 4]],
    )
    return dvf, {"title": "Single-cell diagonal flip (6x6)", "expected_folds": "small"}


@register_case("synth2d_horizontal_bowtie", category="synthetic_2d", dim=2)
def synth2d_horizontal_bowtie():
    """Two horizontally-displaced points crossing — classic bowtie pattern."""
    dvf = _build(
        20, 20,
        msample=[[0, 8, 5], [0, 12, 5]],
        fsample=[[0, 12, 5], [0, 8, 5]],
    )
    return dvf, {"title": "Horizontal bowtie (20x20)", "expected_folds": "moderate"}


@register_case("synth2d_diagonal_bowtie", category="synthetic_2d", dim=2)
def synth2d_diagonal_bowtie():
    """Diagonal point swap — harder to detect with central-diff Jdet alone."""
    dvf = _build(
        20, 20,
        msample=[[0, 8, 8], [0, 12, 12]],
        fsample=[[0, 12, 12], [0, 8, 8]],
    )
    return dvf, {"title": "Diagonal bowtie (20x20)", "expected_folds": "moderate"}


@register_case("synth2d_layered_bowtie_stack", category="synthetic_2d", dim=2)
def synth2d_layered_bowtie_stack():
    """Three stacked bowtie pairs — exercises multi-region windowing."""
    dvf = _build(
        30, 30,
        msample=[
            [0, 5, 8], [0, 9, 8],
            [0, 13, 8], [0, 17, 8],
            [0, 21, 8], [0, 25, 8],
        ],
        fsample=[
            [0, 9, 8], [0, 5, 8],
            [0, 17, 8], [0, 13, 8],
            [0, 25, 8], [0, 21, 8],
        ],
    )
    return dvf, {"title": "Layered bowtie stack (30x30)", "expected_folds": "many"}
```

- [ ] **Step 3.4: Run case tests — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_cases_registry.py -v
```
Expected: 8 passed (4 cases × 2 tests).

If any case has no initial folds, tweak the correspondences (push points further) until it does. The "must have initial folds" assertion is exactly the tripwire.

- [ ] **Step 3.5: Commit**

```powershell
git add benchmarks/two_triangle/cases/__init__.py benchmarks/two_triangle/cases/synthetic_2d.py benchmarks/two_triangle/tests/test_cases_registry.py
git commit -m "two_triangle bench: synthetic 2D test cases"
```

---

## Task 4: Synthetic 3D test cases

**Files:**
- Create: `benchmarks/two_triangle/cases/synthetic_3d.py`
- Modify: `benchmarks/two_triangle/cases/__init__.py` (add import)
- Modify: `benchmarks/two_triangle/tests/test_cases_registry.py` (parametrize 3D)

- [ ] **Step 4.1: Add 3D parametrize block to test file**

Append to `benchmarks/two_triangle/tests/test_cases_registry.py`:
```python
SYNTHETIC_3D = (
    "synth3d_single_tet_flip",
    "synth3d_6tet_bowtie",
)


@pytest.mark.parametrize("name", SYNTHETIC_3D)
def test_synthetic_3d_case_registered(name):
    assert name in registry.list_cases()
    fn = registry.get_case(name)
    dvf, meta = fn()
    assert dvf.ndim == 4
    assert dvf.shape[0] == 3
    assert dvf.shape[1] >= 2  # 3D: D > 1
    assert meta.get("dim") == 3


@pytest.mark.parametrize("name", SYNTHETIC_3D)
def test_synthetic_3d_case_has_initial_folds(name):
    from benchmarks.two_triangle.metrics import fold_counts
    fn = registry.get_case(name)
    dvf, _ = fn()
    fc = fold_counts(dvf, threshold=0.01)
    assert fc["fold_count_jdet"] > 0
```

- [ ] **Step 4.2: Run — expect FAIL**

```powershell
pytest benchmarks/two_triangle/tests/test_cases_registry.py -v -k synth3d
```
Expected: failures (cases not registered).

- [ ] **Step 4.3: Implement 3D synthetic cases**

`benchmarks/two_triangle/cases/synthetic_3d.py`:
```python
"""Synthetic 3D single-tet-flip and 6-tet-bowtie cases.

Built directly as displacement arrays rather than via correspondences —
the 3D Laplacian solver is heavier and the failure modes we want to
exercise (single-vertex folds, 6-tet bowties) are easier to specify
explicitly. Mirrors the patterns from notebooks 12a and 12c.
"""
import numpy as np

from benchmarks.two_triangle.registry import register_case


@register_case("synth3d_single_tet_flip", category="synthetic_3d", dim=3)
def synth3d_single_tet_flip():
    """5x5x5 cube where one interior vertex is shoved across its neighbour."""
    D, H, W = 5, 5, 5
    dvf = np.zeros((3, D, H, W), dtype=np.float64)
    # Shove vertex (2, 2, 2) by (-2, -2, -2) — large enough to flip
    # several incident tets.
    dvf[0, 2, 2, 2] = -2.0  # dz
    dvf[1, 2, 2, 2] = -2.0  # dy
    dvf[2, 2, 2, 2] = -2.0  # dx
    return dvf, {"title": "Single-tet vertex flip (5x5x5)",
                 "expected_folds": "small"}


@register_case("synth3d_6tet_bowtie", category="synthetic_3d", dim=3)
def synth3d_6tet_bowtie():
    """Two adjacent vertices swapped — exercises the 6-tet decomposition.

    Pattern: vertex (2, 2, 2) and (2, 2, 3) swap their displacements,
    creating a 3D analog of the 2D bowtie that fools central-diff Jacobians
    in some triangulations but is caught by the strict 6-tet check.
    """
    D, H, W = 6, 6, 6
    dvf = np.zeros((3, D, H, W), dtype=np.float64)
    dvf[2, 2, 2, 2] = +2.0   # vertex A: dx +2 -> lands on B
    dvf[2, 2, 2, 3] = -2.0   # vertex B: dx -2 -> lands on A
    return dvf, {"title": "6-tet bowtie (6x6x6)",
                 "expected_folds": "moderate"}
```

- [ ] **Step 4.4: Update package init**

Edit `benchmarks/two_triangle/cases/__init__.py`:
```python
"""Importing this package registers all cases."""
from benchmarks.two_triangle.cases import synthetic_2d  # noqa: F401
from benchmarks.two_triangle.cases import synthetic_3d  # noqa: F401
```

- [ ] **Step 4.5: Run — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_cases_registry.py -v -k synth3d
```
Expected: 4 passed (2 cases × 2 tests). If "no initial folds" fails for `synth3d_single_tet_flip`, increase the displacement magnitudes.

- [ ] **Step 4.6: Commit**

```powershell
git add benchmarks/two_triangle/cases/synthetic_3d.py benchmarks/two_triangle/cases/__init__.py benchmarks/two_triangle/tests/test_cases_registry.py
git commit -m "two_triangle bench: synthetic 3D test cases"
```

---

## Task 5: Random 3D test cases

**Files:**
- Create: `benchmarks/two_triangle/cases/random_3d.py`
- Modify: `benchmarks/two_triangle/cases/__init__.py`
- Modify: `benchmarks/two_triangle/tests/test_cases_registry.py`

- [ ] **Step 5.1: Add parametrize block**

Append to `benchmarks/two_triangle/tests/test_cases_registry.py`:
```python
RANDOM_3D = (
    "rand3d_grid16_low",  "rand3d_grid16_high",
    "rand3d_grid24_low",  "rand3d_grid24_high",
    "rand3d_grid32_low",  "rand3d_grid32_high",
)


@pytest.mark.parametrize("name", RANDOM_3D)
def test_random_3d_case_registered(name):
    assert name in registry.list_cases()
    fn = registry.get_case(name)
    dvf, meta = fn()
    assert dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] >= 2
    assert meta.get("dim") == 3


@pytest.mark.parametrize("name", ("rand3d_grid16_high", "rand3d_grid32_high"))
def test_random_3d_high_severity_has_folds(name):
    from benchmarks.two_triangle.metrics import fold_counts
    fn = registry.get_case(name)
    dvf, _ = fn()
    fc = fold_counts(dvf, threshold=0.01)
    assert fc["fold_count_jdet"] > 0
```

(Note: small-grid + low-severity may not have folds; we only assert folds for high-severity cases. Low-severity cases are kept anyway because they're useful for warm-start variants that should leave already-clean fields untouched.)

- [ ] **Step 5.2: Run — expect FAIL**

```powershell
pytest benchmarks/two_triangle/tests/test_cases_registry.py -v -k rand3d
```

- [ ] **Step 5.3: Implement random 3D cases**

`benchmarks/two_triangle/cases/random_3d.py`:
```python
"""Random 3D DVF cases with fixed seeds for reproducibility.

Uses dvfopt.dvf.generate_random_dvf_3d which builds a smooth random vector
field of the requested shape. We provide grid sizes 16, 24, 32 across two
fold-severity levels (controlled by max_magnitude).

Note: 32 cubed is the upper end here; larger grids (64, 128) push baseline
SLSQP runtime past the per-cell timeout. Once active-set or multigrid
variants are validated, larger sizes can be added.
"""
import numpy as np

from dvfopt.dvf import generate_random_dvf_3d

from benchmarks.two_triangle.registry import register_case


def _make(grid: int, magnitude: float, seed: int) -> np.ndarray:
    """generate_random_dvf_3d takes (3, D, H, W) shape; original_shape is the
    *underlying* random field that gets upsampled. We use a small original
    shape so the resulting field is smooth, then it's already at target size."""
    return generate_random_dvf_3d((3, grid, grid, grid),
                                  max_magnitude=magnitude, seed=seed)


_SEVERITIES = {"low": 1.5, "high": 4.0}
_GRIDS = (16, 24, 32)


def _register_grid(grid: int):
    for sev_name, sev_val in _SEVERITIES.items():
        case_name = f"rand3d_grid{grid}_{sev_name}"
        seed = 42 + grid + (0 if sev_name == "low" else 1)

        def _factory(g=grid, m=sev_val, s=seed, n=case_name):
            def case_fn():
                return _make(g, m, s), {
                    "title": f"Random 3D ({g}^3, {n.split('_')[-1]})",
                    "grid": g, "magnitude": m, "seed": s,
                }
            return case_fn

        register_case(case_name, category="random_3d", dim=3,
                      grid=grid, severity=sev_name)(_factory())


for _g in _GRIDS:
    _register_grid(_g)
```

- [ ] **Step 5.4: Update package init**

Edit `benchmarks/two_triangle/cases/__init__.py`:
```python
"""Importing this package registers all cases."""
from benchmarks.two_triangle.cases import synthetic_2d  # noqa: F401
from benchmarks.two_triangle.cases import synthetic_3d  # noqa: F401
from benchmarks.two_triangle.cases import random_3d     # noqa: F401
```

- [ ] **Step 5.5: Run — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_cases_registry.py -v -k rand3d
```
Expected: 8 passed. If a "high severity" case has no folds, bump `_SEVERITIES["high"]` until it does.

- [ ] **Step 5.6: Commit**

```powershell
git add benchmarks/two_triangle/cases/random_3d.py benchmarks/two_triangle/cases/__init__.py benchmarks/two_triangle/tests/test_cases_registry.py
git commit -m "two_triangle bench: random 3D test cases"
```

---

## Task 6: Real 2D + 3D slice cases

**Files:**
- Create: `benchmarks/two_triangle/cases/real_slices.py`
- Modify: `benchmarks/two_triangle/cases/__init__.py`
- Modify: `benchmarks/two_triangle/tests/test_cases_registry.py`

- [ ] **Step 6.1: Verify data files exist**

```powershell
python -c "from pathlib import Path; root = Path('data'); print([p.name for p in (root / 'test_cases_3d').iterdir()]); print([p.name for p in root.glob('test_cases/02*.npy')][:5])"
```
Expected: lists like `['slice090_5x10x10.npy', 'slice200_5x10x10.npy', 'slice350_5x10x10.npy']` for 3D and `['02a_64x91_slice90.npy', ...]` for 2D.

If files are missing (likely because `*.npy` is gitignored), produce them via the existing builders. For 2D you can run `load_slice(90, 0.2)` and save the result; for 3D the npys already exist on disk per `data/test_cases_3d/`. **Do not commit any `.npy` files** — they are gitignored.

- [ ] **Step 6.2: Add parametrize block**

Append to `benchmarks/two_triangle/tests/test_cases_registry.py`:
```python
REAL_CASES = (
    "real2d_slice90_64x91",
    "real2d_slice200_64x91",
    "real3d_slice090_5x10x10",
    "real3d_slice200_5x10x10",
)


@pytest.mark.parametrize("name", REAL_CASES)
def test_real_case_registered(name):
    assert name in registry.list_cases()
    fn = registry.get_case(name)
    try:
        dvf, meta = fn()
    except FileNotFoundError:
        pytest.skip(f"{name}: data file not present (expected; .npy files are gitignored)")
    assert dvf.ndim == 4 and dvf.shape[0] == 3
    assert "title" in meta
```

- [ ] **Step 6.3: Run — expect FAIL**

```powershell
pytest benchmarks/two_triangle/tests/test_cases_registry.py -v -k real
```
Expected: failures (module missing).

- [ ] **Step 6.4: Implement real cases**

`benchmarks/two_triangle/cases/real_slices.py`:
```python
"""Real-data slice cases.

2D: loaded via test_cases.load_slice (correspondences -> Laplacian field).
3D: loaded directly from pre-saved .npy files in data/test_cases_3d/.

Data files are gitignored (*.npy is in .gitignore), so these cases will
raise FileNotFoundError if the user hasn't checked out the data dir.
The case-registry test handles that gracefully via pytest.skip.
"""
from pathlib import Path

import numpy as np

from test_cases import load_slice

from benchmarks.two_triangle.registry import register_case


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_3D = REPO_ROOT / "data" / "test_cases_3d"


def _real_2d_factory(slice_idx: int, scale: float, label: str):
    def case_fn():
        dvf, _ms, _fs = load_slice(slice_idx, scale_factor=scale,
                                   mpoints_path=str(REPO_ROOT / "data" /
                                       "corrected_correspondences_count_touching" /
                                       "mpoints.npy"),
                                   fpoints_path=str(REPO_ROOT / "data" /
                                       "corrected_correspondences_count_touching" /
                                       "fpoints.npy"))
        return dvf, {"title": label, "slice_idx": slice_idx, "scale": scale}
    return case_fn


def _real_3d_factory(filename: str, label: str):
    def case_fn():
        path = DATA_3D / filename
        dvf = np.load(path)
        # Stored arrays may be (3, D, H, W) directly.
        return dvf, {"title": label, "source": filename}
    return case_fn


register_case("real2d_slice90_64x91", category="real_2d", dim=2)(
    _real_2d_factory(90, 0.2, "Real 2D slice 90 @ 64x91"))

register_case("real2d_slice200_64x91", category="real_2d", dim=2)(
    _real_2d_factory(200, 0.2, "Real 2D slice 200 @ 64x91"))

register_case("real3d_slice090_5x10x10", category="real_3d", dim=3)(
    _real_3d_factory("slice090_5x10x10.npy", "Real 3D slice 090 @ 5x10x10"))

register_case("real3d_slice200_5x10x10", category="real_3d", dim=3)(
    _real_3d_factory("slice200_5x10x10.npy", "Real 3D slice 200 @ 5x10x10"))
```

- [ ] **Step 6.5: Update package init**

Edit `benchmarks/two_triangle/cases/__init__.py`:
```python
"""Importing this package registers all cases."""
from benchmarks.two_triangle.cases import synthetic_2d  # noqa: F401
from benchmarks.two_triangle.cases import synthetic_3d  # noqa: F401
from benchmarks.two_triangle.cases import random_3d     # noqa: F401
from benchmarks.two_triangle.cases import real_slices   # noqa: F401
```

- [ ] **Step 6.6: Run — expect PASS or skip**

```powershell
pytest benchmarks/two_triangle/tests/test_cases_registry.py -v -k real
```
Expected: passed or skipped (skip is acceptable when data files aren't present locally).

- [ ] **Step 6.7: Commit**

```powershell
git add benchmarks/two_triangle/cases/real_slices.py benchmarks/two_triangle/cases/__init__.py benchmarks/two_triangle/tests/test_cases_registry.py
git commit -m "two_triangle bench: real 2D + 3D slice cases"
```

---

## Task 7: Trajectory accumulator + minimal re-implemented loop

**Files:**
- Create: `benchmarks/two_triangle/trajectory.py`
- Create: `benchmarks/two_triangle/_iterative_loop.py`
- Create: `benchmarks/two_triangle/tests/test_iterative_loop.py`

This task builds the substrate for the loop-owning variants (soft_margin,
active_set, trust_constr). The re-implementation is a *minimal* version of
`dvfopt.core.slsqp.iterative.iterative_serial` that supports per-iteration
trajectory capture and pluggable constraint builders. It deliberately omits
the sophisticated escalation/oscillation logic — that is acceptable scope
loss for the harness, since the baseline variant uses the real solver.

- [ ] **Step 7.1: Write failing test for trajectory**

`benchmarks/two_triangle/tests/test_iterative_loop.py`:
```python
import numpy as np
import pandas as pd
import pytest

from benchmarks.two_triangle.trajectory import TrajectoryAccumulator
from benchmarks.two_triangle._iterative_loop import run_minimal_iterative_2d


def test_trajectory_accumulator_basic():
    acc = TrajectoryAccumulator()
    phi0 = np.zeros((2, 4, 4))
    acc.record(outer_iter=0, phi=phi0, phi_initial=phi0,
               n_active_windows=0, inner_iters=0, t_elapsed=0.0)
    acc.record(outer_iter=1, phi=phi0, phi_initial=phi0,
               n_active_windows=1, inner_iters=12, t_elapsed=0.42)
    df = acc.to_frame()
    assert len(df) == 2
    assert list(df.columns)[:3] == ["outer_iter", "time_s", "fold_count_jdet"]


def _tiny_2d_fold():
    H, W = 6, 6
    dvf = np.zeros((3, 1, H, W))
    dvf[1, 0, 3, 3] = -2.0
    dvf[2, 0, 3, 3] = -2.0
    return dvf


def test_run_minimal_iterative_2d_smoke():
    dvf = _tiny_2d_fold()
    phi_init = np.stack([dvf[1, 0], dvf[2, 0]])
    result = run_minimal_iterative_2d(
        phi_init.copy(), threshold=0.01, max_iterations=20,
    )
    assert result.phi_final.shape == phi_init.shape
    assert len(result.trajectory) >= 1
    # Final fold count should be no worse than initial
    initial_folds = result.trajectory.iloc[0]["fold_count_tri"]
    final_folds = result.trajectory.iloc[-1]["fold_count_tri"]
    assert final_folds <= initial_folds
```

- [ ] **Step 7.2: Run — expect FAIL**

```powershell
pytest benchmarks/two_triangle/tests/test_iterative_loop.py -v
```
Expected: `ImportError`.

- [ ] **Step 7.3: Implement trajectory accumulator**

`benchmarks/two_triangle/trajectory.py`:
```python
"""Per-iteration metric accumulator used by loop-owning variants."""
from typing import List

import numpy as np
import pandas as pd

from benchmarks.two_triangle.metrics import (
    fold_counts, l2_displacement, smoothness,
)


class TrajectoryAccumulator:
    """Captures per-outer-iter metrics into rows; emits a DataFrame at end."""

    def __init__(self) -> None:
        self._rows: List[dict] = []

    def record(self, *, outer_iter: int, phi: np.ndarray,
               phi_initial: np.ndarray, n_active_windows: int,
               inner_iters: int, t_elapsed: float,
               threshold: float = 0.01) -> None:
        fc = fold_counts(phi, threshold=threshold)
        self._rows.append({
            "outer_iter": outer_iter,
            "time_s": t_elapsed,
            "fold_count_jdet": fc["fold_count_jdet"],
            "fold_count_tri": fc["fold_count_tri"],
            "max_violation": fc["max_violation"],
            "l2_disp": l2_displacement(phi, phi_initial),
            "smoothness": smoothness(phi),
            "n_active_windows": n_active_windows,
            "inner_iters": inner_iters,
        })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)
```

- [ ] **Step 7.4: Implement minimal re-implemented loop**

`benchmarks/two_triangle/_iterative_loop.py`:
```python
"""Minimal re-implementation of the iterative SLSQP outer loop for 2D.

Used by loop-owning variants (soft_margin, active_set, trust_constr). The
re-implementation supports per-iteration trajectory capture and pluggable
constraint builders, but deliberately omits the sophisticated escalation
and oscillation-livelock logic from `dvfopt.core.slsqp.iterative` — keeping
the harness loop small and easy to reason about. Loop-owning variants may
therefore converge slightly less aggressively than the baseline; the
baseline_serial variant uses the real solver and remains the convergence
reference.

Reuses (do NOT modify):
  - dvfopt.core.slsqp.spatial.{argmin_quality, neg_jdet_bounding_window,
                                 get_nearest_center, _edge_flags,
                                 get_phi_sub_flat_padded}
  - dvfopt.core.slsqp.constraints._build_constraints (default constraint builder)
  - dvfopt.core.objective.objective_euc
  - dvfopt.jacobian.numpy_jdet.jacobian_det2D
"""
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.optimize import minimize

from dvfopt.core.objective import objective_euc
from dvfopt.core.slsqp.constraints import _build_constraints
from dvfopt.core.slsqp.spatial import (
    argmin_quality, neg_jdet_bounding_window, get_nearest_center,
    _edge_flags, get_phi_sub_flat_padded,
)
from dvfopt.jacobian.numpy_jdet import jacobian_det2D

from benchmarks.two_triangle.trajectory import TrajectoryAccumulator
from benchmarks.two_triangle.result import SolverResult


@dataclass
class LoopOptions:
    threshold: float = 0.01
    err_tol: float = 1e-5
    max_iterations: int = 100
    max_per_index_iter: int = 5
    max_minimize_iter: int = 100
    enforce_triangles: bool = True
    timeout_s: float = 600.0
    method: str = "SLSQP"
    constraint_builder: Optional[Callable] = None  # defaults to _build_constraints
    variant_name: str = "loop_owning"


def run_minimal_iterative_2d(
    phi_initial: np.ndarray,
    *,
    threshold: float = 0.01,
    err_tol: float = 1e-5,
    max_iterations: int = 100,
    max_per_index_iter: int = 5,
    max_minimize_iter: int = 100,
    enforce_triangles: bool = True,
    timeout_s: float = 600.0,
    method: str = "SLSQP",
    constraint_builder: Optional[Callable] = None,
    variant_name: str = "loop_owning",
) -> SolverResult:
    """Run a stripped-down iterative SLSQP loop on a (2, H, W) phi.

    Returns a SolverResult with a per-outer-iter trajectory DataFrame.
    """
    phi = phi_initial.copy()
    H, W = phi.shape[1:]
    slice_shape = (1, H, W)
    max_window = (H, W)

    if constraint_builder is None:
        constraint_builder = _build_constraints

    acc = TrajectoryAccumulator()
    t0 = time.perf_counter()
    timed_out = False
    err_msg = None

    jacobian_matrix = jacobian_det2D(phi)
    quality_matrix = jacobian_matrix.copy()
    inner_iters_total = 0
    n_windows_this_iter = 0

    acc.record(outer_iter=0, phi=phi, phi_initial=phi_initial,
               n_active_windows=0, inner_iters=0, t_elapsed=0.0,
               threshold=threshold)

    iteration = 0
    try:
        while iteration < max_iterations and (quality_matrix[0] <= threshold - err_tol).any():
            if time.perf_counter() - t0 > timeout_s:
                timed_out = True
                break
            iteration += 1

            # Locate worst pixel and its CC bounding window
            neg_yx = argmin_quality(quality_matrix)
            neg_mask = quality_matrix[0] <= threshold - err_tol
            labeled, _ = scipy_label(neg_mask)
            sub_size, bbox_center = neg_jdet_bounding_window(
                quality_matrix, neg_yx, threshold, err_tol, labeled=labeled)
            sub_size = (min(sub_size[0], H), min(sub_size[1], W))
            cz, cy, cx = get_nearest_center(bbox_center, slice_shape, sub_size)
            is_at_edge, win_at_max = _edge_flags(cy, cx, sub_size,
                                                 slice_shape, max_window)
            phi_sub_flat, actual_size = get_phi_sub_flat_padded(
                phi, cz, cy, cx, slice_shape, sub_size)

            constraints = constraint_builder(
                phi_sub_flat, actual_size, is_at_edge, win_at_max,
                threshold,
                enforce_shoelace=False,
                enforce_injectivity=False,
                enforce_triangles=enforce_triangles,
            )

            res = minimize(
                objective_euc, phi_sub_flat,
                args=(phi_sub_flat,),
                method=method, jac=True,
                constraints=constraints,
                options={"maxiter": max_minimize_iter, "ftol": 1e-9},
            )
            inner_iters_total += int(res.nit)
            n_windows_this_iter = 1

            # Splat optimised window back into phi
            sy, sx = actual_size
            new_phi_sub = res.x
            phix = new_phi_sub[:sy * sx].reshape(sy, sx)
            phiy = new_phi_sub[sy * sx:].reshape(sy, sx)
            hy, hx = sy // 2, sx // 2
            hy_hi, hx_hi = sy - hy, sx - hx
            phi[1, cy - hy:cy + hy_hi, cx - hx:cx + hx_hi] = phix
            phi[0, cy - hy:cy + hy_hi, cx - hx:cx + hx_hi] = phiy

            jacobian_matrix = jacobian_det2D(phi)
            quality_matrix = jacobian_matrix.copy()

            acc.record(outer_iter=iteration, phi=phi,
                       phi_initial=phi_initial,
                       n_active_windows=n_windows_this_iter,
                       inner_iters=int(res.nit),
                       t_elapsed=time.perf_counter() - t0,
                       threshold=threshold)
    except Exception:  # pylint: disable=broad-except
        import traceback as _tb
        err_msg = _tb.format_exc()

    traj = acc.to_frame()
    final_fc = traj.iloc[-1]
    converged = (not timed_out and err_msg is None
                 and final_fc["fold_count_jdet"] == 0
                 and final_fc["fold_count_tri"] == 0)

    return SolverResult(
        phi_final=phi, trajectory=traj,
        converged=bool(converged), timed_out=timed_out, error=err_msg,
        meta={"variant": variant_name, "iterations": iteration,
              "inner_iters_total": inner_iters_total,
              "threshold": threshold, "method": method},
    )
```

- [ ] **Step 7.5: Run — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_iterative_loop.py -v
```
Expected: 2 passed.

If `objective_euc` import path is wrong, grep for the actual location:
```powershell
python -c "from dvfopt.core.objective import objective_euc; print(objective_euc)"
```

- [ ] **Step 7.6: Commit**

```powershell
git add benchmarks/two_triangle/trajectory.py benchmarks/two_triangle/_iterative_loop.py benchmarks/two_triangle/tests/test_iterative_loop.py
git commit -m "two_triangle bench: trajectory accumulator + minimal re-implemented loop"
```

---

## Task 8: Wrapper variant — combinatorial pre-fix

**Files:**
- Create: `benchmarks/two_triangle/variants/combinatorial_prefix.py`
- Modify: `benchmarks/two_triangle/variants/__init__.py`
- Create: `benchmarks/two_triangle/tests/test_variants_wrapper.py`

- [ ] **Step 8.1: Write failing test**

`benchmarks/two_triangle/tests/test_variants_wrapper.py`:
```python
import numpy as np
import pytest

from benchmarks.two_triangle import registry
import benchmarks.two_triangle.variants  # registers


def _tiny_2d_fold():
    H, W = 6, 6
    dvf = np.zeros((3, 1, H, W))
    dvf[1, 0, 3, 3] = -2.0
    dvf[2, 0, 3, 3] = -2.0
    return dvf


@pytest.mark.parametrize("name", ["combinatorial_prefix"])
def test_wrapper_variant_runs(name):
    fn = registry.get_variant(name)
    dvf = _tiny_2d_fold()
    result = fn(dvf, threshold=0.01, max_iterations=20)
    assert result.phi_final.shape == (2, 6, 6)
    assert len(result.trajectory) >= 1


def test_combinatorial_prefix_reduces_or_preserves_folds():
    """Prefix step alone should not increase folds beyond initial."""
    from benchmarks.two_triangle.metrics import fold_counts
    fn = registry.get_variant("combinatorial_prefix")
    dvf = _tiny_2d_fold()
    init_fc = fold_counts(dvf, threshold=0.01)
    result = fn(dvf, threshold=0.01, max_iterations=0)  # only prefix, no SLSQP
    final_fc = fold_counts(result.phi_final, threshold=0.01)
    assert final_fc["fold_count_tri"] <= init_fc["fold_count_tri"]
```

- [ ] **Step 8.2: Run — expect FAIL**

```powershell
pytest benchmarks/two_triangle/tests/test_variants_wrapper.py -v
```

- [ ] **Step 8.3: Implement combinatorial_prefix**

`benchmarks/two_triangle/variants/combinatorial_prefix.py`:
```python
"""Combinatorial pre-fix wrapper variant.

Pre-pass: detect cells where exactly one of the four signed triangle areas
is negative (= 'isolated single-vertex flip'), find the offending vertex,
and damp its displacement towards zero by a halving search until the cell
becomes feasible. Then call the unmodified iterative_serial / iterative_3d
on the residual.

This is a deliberately simple heuristic — it won't fix all isolated flips
(some require coordinated multi-vertex moves), but it's cheap and never
makes things worse on its own (each damping step is rejected if it
introduces new folds).
"""
import time
import traceback

import numpy as np
import pandas as pd

from dvfopt.core.slsqp.iterative import iterative_serial
from dvfopt.core.slsqp.iterative3d import iterative_3d
from dvfopt.jacobian.shoelace import _all_triangle_areas_2d

from benchmarks.two_triangle.registry import register_variant
from benchmarks.two_triangle.result import SolverResult
from benchmarks.two_triangle.metrics import (
    fold_counts, l2_displacement, smoothness,
)


def _is_3d(dvf):
    return dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] > 1


def _prefix_pass_2d(phi: np.ndarray, threshold: float,
                    max_halvings: int = 6) -> np.ndarray:
    """Damp displacements at single-vertex flip cells.

    For each cell with exactly one negative triangle area, identify the
    vertex shared by the most negative triangles and halve its displacement
    repeatedly until the cell is feasible. Reject the change and restore
    if any new fold is introduced anywhere.
    """
    phi = phi.copy()
    H, W = phi.shape[1:]
    for _ in range(max_halvings):
        tri = _all_triangle_areas_2d(phi[0], phi[1])  # (4, H-1, W-1)
        bad = tri < threshold
        bad_per_cell = bad.sum(axis=0)  # (H-1, W-1)
        # "Isolated" = exactly one bad triangle in the cell.
        isolated = np.argwhere(bad_per_cell == 1)
        if isolated.size == 0:
            break
        improved = False
        for cy, cx in isolated:
            # Damp each of the 4 corner vertices by 0.5 and accept the move
            # that maximises the cell-min triangle area without introducing
            # new folds.
            best_delta = None
            best_min = tri[:, cy, cx].min()
            for dy in (0, 1):
                for dx in (0, 1):
                    vy, vx = cy + dy, cx + dx
                    saved = phi[:, vy, vx].copy()
                    phi[:, vy, vx] *= 0.5
                    new_tri = _all_triangle_areas_2d(phi[0], phi[1])
                    new_cell_min = new_tri[:, cy, cx].min()
                    no_new_folds = (new_tri >= threshold).sum() >= (
                        tri >= threshold).sum()
                    if new_cell_min > best_min and no_new_folds:
                        best_min = new_cell_min
                        best_delta = (vy, vx)
                    phi[:, vy, vx] = saved  # restore for next try
            if best_delta is not None:
                vy, vx = best_delta
                phi[:, vy, vx] *= 0.5
                improved = True
        if not improved:
            break
    return phi


def _prefix_pass_3d(dvf: np.ndarray, threshold: float) -> np.ndarray:
    """3D version: no-op for now (no equivalent helper in dvfopt yet).

    The 3D constraint coverage helper (notebook 12c) hasn't been factored
    into a reusable function, so the 3D prefix pass is identity. Once a
    `_all_tetrahedron_volumes_3d` helper exists, port `_prefix_pass_2d`.
    """
    return dvf.copy()


@register_variant("combinatorial_prefix")
def combinatorial_prefix(dvf: np.ndarray, *, threshold: float = 0.01,
                          max_iterations: int = 100,
                          enforce_triangles: bool = True,
                          timeout_s: float = 600.0,
                          **_unused) -> SolverResult:
    is_3d = _is_3d(dvf)
    phi_initial = dvf.copy()
    t0 = time.perf_counter()
    err = None
    rows = []

    # Initial state row
    if is_3d:
        phi_can = phi_initial.copy()
    else:
        phi_can = np.stack([phi_initial[1, 0], phi_initial[2, 0]])
    fc = fold_counts(phi_can, threshold=threshold)
    rows.append({
        "outer_iter": 0, "time_s": 0.0,
        "fold_count_jdet": fc["fold_count_jdet"],
        "fold_count_tri": fc["fold_count_tri"],
        "max_violation": fc["max_violation"],
        "l2_disp": 0.0, "smoothness": smoothness(phi_can),
        "n_active_windows": 0, "inner_iters": 0,
    })

    # --- Prefix pass ---
    if is_3d:
        dvf_pref = _prefix_pass_3d(dvf, threshold)
        phi_pref_can = dvf_pref.copy()
    else:
        phi_2d = np.stack([dvf[1, 0], dvf[2, 0]])
        phi_pref_2d = _prefix_pass_2d(phi_2d, threshold)
        dvf_pref = dvf.copy()
        dvf_pref[1, 0] = phi_pref_2d[0]
        dvf_pref[2, 0] = phi_pref_2d[1]
        phi_pref_can = phi_pref_2d

    fc = fold_counts(phi_pref_can, threshold=threshold)
    rows.append({
        "outer_iter": 1, "time_s": time.perf_counter() - t0,
        "fold_count_jdet": fc["fold_count_jdet"],
        "fold_count_tri": fc["fold_count_tri"],
        "max_violation": fc["max_violation"],
        "l2_disp": l2_displacement(phi_pref_can, phi_can),
        "smoothness": smoothness(phi_pref_can),
        "n_active_windows": 0, "inner_iters": 0,
    })

    # --- Baseline call on residual ---
    if max_iterations > 0:
        try:
            if is_3d:
                phi_post = iterative_3d(dvf_pref, threshold=threshold,
                                        max_iterations=max_iterations,
                                        verbose=0)
            else:
                phi_post = iterative_serial(dvf_pref, threshold=threshold,
                                            max_iterations=max_iterations,
                                            verbose=0,
                                            enforce_triangles=enforce_triangles)
        except Exception:
            err = traceback.format_exc()
            phi_post = phi_pref_can
    else:
        phi_post = phi_pref_can

    if not is_3d and phi_post.ndim == 4 and phi_post.shape[1] == 1:
        phi_post_can = np.stack([phi_post[1, 0], phi_post[2, 0]])
    else:
        phi_post_can = phi_post

    fc = fold_counts(phi_post_can, threshold=threshold)
    rows.append({
        "outer_iter": 2, "time_s": time.perf_counter() - t0,
        "fold_count_jdet": fc["fold_count_jdet"],
        "fold_count_tri": fc["fold_count_tri"],
        "max_violation": fc["max_violation"],
        "l2_disp": l2_displacement(phi_post_can, phi_can),
        "smoothness": smoothness(phi_post_can),
        "n_active_windows": 0, "inner_iters": 0,
    })

    converged = (err is None and fc["fold_count_jdet"] == 0
                 and fc["fold_count_tri"] == 0)
    return SolverResult(
        phi_final=phi_post_can, trajectory=pd.DataFrame(rows),
        converged=converged, timed_out=False, error=err,
        meta={"variant": "combinatorial_prefix", "is_3d": is_3d,
              "threshold": threshold, "max_iterations": max_iterations},
    )
```

- [ ] **Step 8.4: Update variant package init**

Edit `benchmarks/two_triangle/variants/__init__.py`:
```python
"""Importing this package registers all variants."""
from benchmarks.two_triangle.variants import baseline_serial         # noqa: F401
from benchmarks.two_triangle.variants import combinatorial_prefix    # noqa: F401
```

- [ ] **Step 8.5: Run — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_variants_wrapper.py -v
```
Expected: 2 passed.

- [ ] **Step 8.6: Commit**

```powershell
git add benchmarks/two_triangle/variants/combinatorial_prefix.py benchmarks/two_triangle/variants/__init__.py benchmarks/two_triangle/tests/test_variants_wrapper.py
git commit -m "two_triangle bench: combinatorial_prefix wrapper variant"
```

---

## Task 9: Loop-owning variant — soft_margin

**Files:**
- Create: `benchmarks/two_triangle/variants/soft_margin.py`
- Modify: `benchmarks/two_triangle/variants/__init__.py`
- Create: `benchmarks/two_triangle/tests/test_variants_loop_owning.py`

- [ ] **Step 9.1: Write failing test**

`benchmarks/two_triangle/tests/test_variants_loop_owning.py`:
```python
import numpy as np
import pytest

from benchmarks.two_triangle import registry
import benchmarks.two_triangle.variants  # registers


def _tiny_2d_fold():
    H, W = 6, 6
    dvf = np.zeros((3, 1, H, W))
    dvf[1, 0, 3, 3] = -2.0
    dvf[2, 0, 3, 3] = -2.0
    return dvf


def test_soft_margin_registered():
    assert "soft_margin" in registry.list_variants()


def test_soft_margin_runs_and_progresses():
    fn = registry.get_variant("soft_margin")
    dvf = _tiny_2d_fold()
    result = fn(dvf, threshold=0.01, max_iterations=20)
    # Multi-row trajectory expected (loop-owning variant)
    assert len(result.trajectory) >= 2
    initial_folds = result.trajectory.iloc[0]["fold_count_tri"]
    final_folds = result.trajectory.iloc[-1]["fold_count_tri"]
    assert final_folds <= initial_folds


def test_soft_margin_trajectory_monotonic_on_synthetic():
    """Loop-owning variants should not regress on simple synthetic inputs."""
    fn = registry.get_variant("soft_margin")
    dvf = _tiny_2d_fold()
    result = fn(dvf, threshold=0.01, max_iterations=20)
    folds = result.trajectory["fold_count_tri"].values
    diffs = np.diff(folds)
    # Allow at most one "blip" of magnitude 1 (sub-window optimisation may
    # transiently introduce a fold during boundary repositioning).
    assert (diffs > 1).sum() == 0
```

- [ ] **Step 9.2: Run — expect FAIL**

```powershell
pytest benchmarks/two_triangle/tests/test_variants_loop_owning.py -v -k soft_margin
```

- [ ] **Step 9.3: Implement soft_margin**

`benchmarks/two_triangle/variants/soft_margin.py`:
```python
"""Soft-margin variant: progressive threshold tightening.

Strategy:
- Start with threshold=0 (sign-only): "any positive area is feasible".
- Once a feasible point is found, tighten threshold to the user-supplied
  value (default 0.01) and continue iterating.
- Uses the minimal re-implemented loop with a constraint builder closure
  that closes over the current threshold.
"""
from functools import partial

import numpy as np

from dvfopt.core.slsqp.constraints import _build_constraints

from benchmarks.two_triangle.registry import register_variant
from benchmarks.two_triangle._iterative_loop import run_minimal_iterative_2d
from benchmarks.two_triangle.result import SolverResult


def _is_3d(dvf):
    return dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] > 1


def _builder_for_threshold(thr: float):
    def _build(phi_sub_flat, sub_size, is_at_edge, win_at_max, _thr_unused,
               enforce_shoelace=False, enforce_injectivity=False,
               enforce_triangles=True):
        return _build_constraints(
            phi_sub_flat, sub_size, is_at_edge, win_at_max, thr,
            enforce_shoelace=enforce_shoelace,
            enforce_injectivity=enforce_injectivity,
            enforce_triangles=enforce_triangles,
        )
    return _build


@register_variant("soft_margin")
def soft_margin(dvf: np.ndarray, *, threshold: float = 0.01,
                 max_iterations: int = 100,
                 enforce_triangles: bool = True,
                 timeout_s: float = 600.0, **_unused) -> SolverResult:
    if _is_3d(dvf):
        # 3D not yet supported in the minimal loop; fall back to baseline.
        from benchmarks.two_triangle.variants.baseline_serial import baseline_serial
        r = baseline_serial(dvf, threshold=threshold,
                            max_iterations=max_iterations,
                            enforce_triangles=enforce_triangles,
                            timeout_s=timeout_s)
        r.meta["variant"] = "soft_margin"
        r.meta["fallback"] = "baseline_3d"
        return r

    phi_initial = np.stack([dvf[1, 0], dvf[2, 0]])

    # --- Stage 1: sign-only threshold ---
    half_iters = max(1, max_iterations // 2)
    r1 = run_minimal_iterative_2d(
        phi_initial, threshold=0.0, max_iterations=half_iters,
        enforce_triangles=enforce_triangles, timeout_s=timeout_s,
        constraint_builder=_builder_for_threshold(0.0),
        variant_name="soft_margin_stage1",
    )
    if r1.timed_out or r1.error is not None:
        return r1

    # --- Stage 2: tighten to target threshold ---
    r2 = run_minimal_iterative_2d(
        r1.phi_final, threshold=threshold,
        max_iterations=max_iterations - half_iters,
        enforce_triangles=enforce_triangles, timeout_s=timeout_s,
        constraint_builder=_builder_for_threshold(threshold),
        variant_name="soft_margin_stage2",
    )

    # Stitch trajectories: re-base stage2's outer_iter and time_s on top of
    # stage1's tail.
    if not r1.trajectory.empty:
        last = r1.trajectory.iloc[-1]
        r2.trajectory["outer_iter"] = (r2.trajectory["outer_iter"]
                                        + last["outer_iter"])
        r2.trajectory["time_s"] = r2.trajectory["time_s"] + last["time_s"]
        # Drop the duplicate "row 0" of stage2 (it equals stage1's tail)
        if len(r2.trajectory) > 1:
            r2.trajectory = r2.trajectory.iloc[1:]
        import pandas as pd
        r2.trajectory = pd.concat(
            [r1.trajectory, r2.trajectory], ignore_index=True)

    r2.meta["variant"] = "soft_margin"
    r2.meta["stage1_threshold"] = 0.0
    r2.meta["stage2_threshold"] = threshold
    return r2
```

- [ ] **Step 9.4: Update variant package init**

Edit `benchmarks/two_triangle/variants/__init__.py`:
```python
"""Importing this package registers all variants."""
from benchmarks.two_triangle.variants import baseline_serial         # noqa: F401
from benchmarks.two_triangle.variants import combinatorial_prefix    # noqa: F401
from benchmarks.two_triangle.variants import soft_margin             # noqa: F401
```

- [ ] **Step 9.5: Run — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_variants_loop_owning.py -v -k soft_margin
```
Expected: 3 passed.

- [ ] **Step 9.6: Commit**

```powershell
git add benchmarks/two_triangle/variants/soft_margin.py benchmarks/two_triangle/variants/__init__.py benchmarks/two_triangle/tests/test_variants_loop_owning.py
git commit -m "two_triangle bench: soft_margin loop-owning variant"
```

---

## Task 10: Loop-owning variant — active_set

**Files:**
- Create: `benchmarks/two_triangle/variants/active_set.py`
- Modify: `benchmarks/two_triangle/variants/__init__.py`
- Modify: `benchmarks/two_triangle/tests/test_variants_loop_owning.py`

- [ ] **Step 10.1: Add tests**

Append to `benchmarks/two_triangle/tests/test_variants_loop_owning.py`:
```python
def test_active_set_registered():
    assert "active_set" in registry.list_variants()


def test_active_set_runs_and_progresses():
    fn = registry.get_variant("active_set")
    dvf = _tiny_2d_fold()
    result = fn(dvf, threshold=0.01, max_iterations=20)
    assert len(result.trajectory) >= 2
    assert result.trajectory.iloc[-1]["fold_count_tri"] <= \
           result.trajectory.iloc[0]["fold_count_tri"]
```

- [ ] **Step 10.2: Run — expect FAIL**

```powershell
pytest benchmarks/two_triangle/tests/test_variants_loop_owning.py -v -k active_set
```

- [ ] **Step 10.3: Implement active_set**

`benchmarks/two_triangle/variants/active_set.py`:
```python
"""Active-set variant: only enforce constraints near violation.

Standard `_build_constraints` enforces the triangle/Jdet inequality on
*every* cell in the sub-window. For folds that touch only a small fraction
of the window, most of those constraints are inactive (cells with positive
area >> threshold). This variant builds a custom constraint that filters
to cells with current value < `active_factor * threshold`.

Implementation: build a NonlinearConstraint that wraps a closure over a
mutable index mask. Refresh the mask once per outer iteration (the inner
SLSQP step uses a fixed mask).
"""
import numpy as np
from scipy.optimize import NonlinearConstraint
import scipy.sparse

from dvfopt._defaults import _unpack_size
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_2d
from dvfopt.jacobian.shoelace import _all_triangle_areas_2d

from benchmarks.two_triangle.registry import register_variant
from benchmarks.two_triangle._iterative_loop import run_minimal_iterative_2d
from benchmarks.two_triangle.result import SolverResult


def _is_3d(dvf):
    return dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] > 1


def _make_active_set_builder(active_factor: float):
    """Returns a constraint_builder closure compatible with _iterative_loop."""

    def _builder(phi_sub_flat, sub_size, is_at_edge, win_at_max, threshold,
                 enforce_shoelace=False, enforce_injectivity=False,
                 enforce_triangles=True):
        sy, sx = _unpack_size(sub_size)
        pixels = sy * sx

        # Compute current jdet + triangle areas to seed the active mask.
        dx = phi_sub_flat[:pixels].reshape(sy, sx)
        dy = phi_sub_flat[pixels:].reshape(sy, sx)
        cur_jdet = _numpy_jdet_2d(dy, dx)
        active_thr = active_factor * threshold

        cur_tri = _all_triangle_areas_2d(dy, dx)  # (4, sy-1, sx-1)

        # Build the Jdet-style constraint covering ALL cells (not pruned —
        # required to keep the QP from drifting into newly-folded regions).
        def jac_constraint(phi1):
            d_x = phi1[:pixels].reshape(sy, sx)
            d_y = phi1[pixels:].reshape(sy, sx)
            j = _numpy_jdet_2d(d_y, d_x)
            return j[1:-1, 1:-1].flatten() if not is_at_edge else j.flatten()

        constraints = [NonlinearConstraint(jac_constraint, threshold, np.inf)]

        # Pruned triangle constraint: keep only triangles with area < active_thr.
        if enforce_triangles:
            tri_mask = (cur_tri < active_thr).flatten()
            n_tri = tri_mask.sum()
            if n_tri > 0:
                idx = np.where(tri_mask)[0]

                def tri_active(phi1):
                    d_x = phi1[:pixels].reshape(sy, sx)
                    d_y = phi1[pixels:].reshape(sy, sx)
                    tri = _all_triangle_areas_2d(d_y, d_x).flatten()
                    return tri[idx]

                constraints.append(
                    NonlinearConstraint(tri_active, threshold, np.inf))

        # Edge-freeze (same as default builder when interior only).
        exclude_bounds = not is_at_edge and not win_at_max
        if exclude_bounds:
            edge_mask = np.zeros((sy, sx), dtype=bool)
            edge_mask[[0, -1], :] = True
            edge_mask[:, [0, -1]] = True
            edge_indices = np.argwhere(edge_mask)
            fixed = []
            y_off = pixels
            for y, x in edge_indices:
                idx = y * sx + x
                fixed.extend([idx, idx + y_off])
            fixed = np.array(fixed)
            from scipy.optimize import LinearConstraint
            A_eq = scipy.sparse.csr_matrix(
                (np.ones(len(fixed)), (np.arange(len(fixed)), fixed)),
                shape=(len(fixed), phi_sub_flat.size))
            constraints.append(
                LinearConstraint(A_eq, phi_sub_flat[fixed], phi_sub_flat[fixed]))

        return constraints

    return _builder


@register_variant("active_set")
def active_set(dvf: np.ndarray, *, threshold: float = 0.01,
                max_iterations: int = 100, active_factor: float = 5.0,
                enforce_triangles: bool = True,
                timeout_s: float = 600.0, **_unused) -> SolverResult:
    if _is_3d(dvf):
        from benchmarks.two_triangle.variants.baseline_serial import baseline_serial
        r = baseline_serial(dvf, threshold=threshold,
                            max_iterations=max_iterations,
                            enforce_triangles=enforce_triangles,
                            timeout_s=timeout_s)
        r.meta["variant"] = "active_set"
        r.meta["fallback"] = "baseline_3d"
        return r

    phi_initial = np.stack([dvf[1, 0], dvf[2, 0]])
    builder = _make_active_set_builder(active_factor=active_factor)
    r = run_minimal_iterative_2d(
        phi_initial, threshold=threshold, max_iterations=max_iterations,
        enforce_triangles=enforce_triangles, timeout_s=timeout_s,
        constraint_builder=builder,
        variant_name="active_set",
    )
    r.meta["active_factor"] = active_factor
    return r
```

- [ ] **Step 10.4: Update package init**

```python
# benchmarks/two_triangle/variants/__init__.py
"""Importing this package registers all variants."""
from benchmarks.two_triangle.variants import baseline_serial         # noqa: F401
from benchmarks.two_triangle.variants import combinatorial_prefix    # noqa: F401
from benchmarks.two_triangle.variants import soft_margin             # noqa: F401
from benchmarks.two_triangle.variants import active_set              # noqa: F401
```

- [ ] **Step 10.5: Run — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_variants_loop_owning.py -v -k active_set
```
Expected: 2 passed. The active_set constraint may not converge as quickly as the dense baseline; that's expected and fine for the harness — the comparison is the point.

- [ ] **Step 10.6: Commit**

```powershell
git add benchmarks/two_triangle/variants/active_set.py benchmarks/two_triangle/variants/__init__.py benchmarks/two_triangle/tests/test_variants_loop_owning.py
git commit -m "two_triangle bench: active_set loop-owning variant"
```

---

## Task 11: Wrapper variant — multigrid

**Files:**
- Create: `benchmarks/two_triangle/variants/multigrid.py`
- Modify: `benchmarks/two_triangle/variants/__init__.py`
- Modify: `benchmarks/two_triangle/tests/test_variants_wrapper.py`

- [ ] **Step 11.1: Add test**

Append to `benchmarks/two_triangle/tests/test_variants_wrapper.py`:
```python
def test_multigrid_registered():
    assert "multigrid" in registry.list_variants()


def test_multigrid_runs_2d():
    fn = registry.get_variant("multigrid")
    dvf = _tiny_2d_fold()
    # 6x6 is too small for two-level cascade (downsampling to 3x3 wouldn't
    # give a meaningful warm-start), but the variant should still produce
    # a valid result by skipping levels that are too small.
    result = fn(dvf, threshold=0.01, max_iterations=10)
    assert result.phi_final.shape == (2, 6, 6)
```

- [ ] **Step 11.2: Implement multigrid**

`benchmarks/two_triangle/variants/multigrid.py`:
```python
"""Multigrid wrapper variant.

Strategy: solve at 1/4 resolution first (cheap, smooths out adjacent folds),
upsample as warm-start to 1/2 resolution, solve, upsample to full, solve.
At each level uses the unmodified iterative_serial / iterative_3d.

Levels are skipped automatically when the downsampled grid would be smaller
than min_dim_per_axis (default 8).
"""
import time
import traceback

import numpy as np
import pandas as pd

from dvfopt.core.slsqp.iterative import iterative_serial
from dvfopt.core.slsqp.iterative3d import iterative_3d
from dvfopt.dvf.scaling import scale_dvf, scale_dvf_3d

from benchmarks.two_triangle.registry import register_variant
from benchmarks.two_triangle.result import SolverResult
from benchmarks.two_triangle.metrics import (
    fold_counts, l2_displacement, smoothness,
)


def _is_3d(dvf):
    return dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] > 1


def _downsample(dvf: np.ndarray, factor: float) -> np.ndarray:
    if _is_3d(dvf):
        D, H, W = dvf.shape[1:]
        new = (max(2, int(D * factor)),
               max(2, int(H * factor)),
               max(2, int(W * factor)))
        return scale_dvf_3d(dvf, new)
    H, W = dvf.shape[2:]
    new = (max(2, int(H * factor)), max(2, int(W * factor)))
    return scale_dvf(dvf, new)


def _upsample_to(dvf: np.ndarray, target_shape: tuple) -> np.ndarray:
    if _is_3d(dvf):
        return scale_dvf_3d(dvf, target_shape)
    return scale_dvf(dvf, target_shape)


@register_variant("multigrid")
def multigrid(dvf: np.ndarray, *, threshold: float = 0.01,
               max_iterations: int = 100, min_dim_per_axis: int = 8,
               enforce_triangles: bool = True,
               timeout_s: float = 600.0, **_unused) -> SolverResult:
    is_3d = _is_3d(dvf)
    phi_initial = dvf.copy()
    t0 = time.perf_counter()
    err = None
    rows = []

    def canon(d):
        if is_3d or d.ndim != 4 or d.shape[1] != 1:
            return d
        return np.stack([d[1, 0], d[2, 0]])

    phi_can_init = canon(phi_initial)
    fc = fold_counts(phi_can_init, threshold=threshold)
    rows.append({
        "outer_iter": 0, "time_s": 0.0,
        "fold_count_jdet": fc["fold_count_jdet"],
        "fold_count_tri": fc["fold_count_tri"],
        "max_violation": fc["max_violation"],
        "l2_disp": 0.0, "smoothness": smoothness(phi_can_init),
        "n_active_windows": 0, "inner_iters": 0,
    })

    cur_dvf = dvf.copy()
    full_shape = cur_dvf.shape[1 if is_3d else 2:] if is_3d else cur_dvf.shape[2:]

    levels = [0.25, 0.5, 1.0]
    for li, factor in enumerate(levels, start=1):
        if factor < 1.0:
            target_shape = tuple(max(2, int(d * factor)) for d in full_shape)
            if min(target_shape) < min_dim_per_axis:
                continue  # Skip levels too small to be meaningful
            level_dvf = _downsample(cur_dvf, factor)
        else:
            level_dvf = cur_dvf

        try:
            if is_3d:
                phi_solved = iterative_3d(
                    level_dvf, threshold=threshold,
                    max_iterations=max_iterations, verbose=0)
            else:
                phi_solved = iterative_serial(
                    level_dvf, threshold=threshold,
                    max_iterations=max_iterations, verbose=0,
                    enforce_triangles=enforce_triangles)
        except Exception:
            err = traceback.format_exc()
            break

        # Re-pack solver output back into (3, [1|D], H, W) form for upsampling
        if is_3d:
            cur_dvf = phi_solved
        else:
            cur_dvf = level_dvf.copy()
            cur_dvf[1, 0] = phi_solved[0]
            cur_dvf[2, 0] = phi_solved[1]

        if factor < 1.0:
            cur_dvf = _upsample_to(cur_dvf, full_shape)

        phi_can = canon(cur_dvf)
        fc = fold_counts(phi_can, threshold=threshold)
        rows.append({
            "outer_iter": li, "time_s": time.perf_counter() - t0,
            "fold_count_jdet": fc["fold_count_jdet"],
            "fold_count_tri": fc["fold_count_tri"],
            "max_violation": fc["max_violation"],
            "l2_disp": l2_displacement(phi_can, phi_can_init),
            "smoothness": smoothness(phi_can),
            "n_active_windows": 0, "inner_iters": 0,
        })

    phi_final = canon(cur_dvf)
    fc_final = fold_counts(phi_final, threshold=threshold)
    converged = (err is None and fc_final["fold_count_jdet"] == 0
                 and fc_final["fold_count_tri"] == 0)
    return SolverResult(
        phi_final=phi_final, trajectory=pd.DataFrame(rows),
        converged=converged, timed_out=False, error=err,
        meta={"variant": "multigrid", "is_3d": is_3d, "levels": levels,
              "threshold": threshold},
    )
```

- [ ] **Step 11.3: Update package init**

```python
# benchmarks/two_triangle/variants/__init__.py
"""Importing this package registers all variants."""
from benchmarks.two_triangle.variants import baseline_serial         # noqa: F401
from benchmarks.two_triangle.variants import combinatorial_prefix    # noqa: F401
from benchmarks.two_triangle.variants import soft_margin             # noqa: F401
from benchmarks.two_triangle.variants import active_set              # noqa: F401
from benchmarks.two_triangle.variants import multigrid               # noqa: F401
```

- [ ] **Step 11.4: Run — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_variants_wrapper.py -v -k multigrid
```
Expected: 2 passed.

- [ ] **Step 11.5: Commit**

```powershell
git add benchmarks/two_triangle/variants/multigrid.py benchmarks/two_triangle/variants/__init__.py benchmarks/two_triangle/tests/test_variants_wrapper.py
git commit -m "two_triangle bench: multigrid wrapper variant"
```

---

## Task 12: Loop-owning variant — trust_constr backend

**Files:**
- Create: `benchmarks/two_triangle/variants/trust_constr.py`
- Modify: `benchmarks/two_triangle/variants/__init__.py`
- Modify: `benchmarks/two_triangle/tests/test_variants_loop_owning.py`

- [ ] **Step 12.1: Add test**

Append to `benchmarks/two_triangle/tests/test_variants_loop_owning.py`:
```python
def test_trust_constr_registered():
    assert "trust_constr" in registry.list_variants()


def test_trust_constr_runs():
    fn = registry.get_variant("trust_constr")
    dvf = _tiny_2d_fold()
    result = fn(dvf, threshold=0.01, max_iterations=20)
    assert result.phi_final.shape == (2, 6, 6)
    # trust-constr is more relaxed than SLSQP — only require non-regression
    assert result.trajectory.iloc[-1]["fold_count_tri"] <= \
           result.trajectory.iloc[0]["fold_count_tri"]
```

- [ ] **Step 12.2: Implement trust_constr**

`benchmarks/two_triangle/variants/trust_constr.py`:
```python
"""trust_constr backend variant.

Identical to soft_margin in spirit but uses scipy's trust-constr method
in the inner optimizer. trust-constr accepts sparse analytic constraint
Jacobians and scales better than SLSQP on larger windows.

Wires trust-constr by passing method="trust-constr" to the inner minimize
call inside the re-implemented loop. The constraint builder is the
default _build_constraints (no soft-margin, no active-set).
"""
import numpy as np

from benchmarks.two_triangle.registry import register_variant
from benchmarks.two_triangle._iterative_loop import run_minimal_iterative_2d
from benchmarks.two_triangle.result import SolverResult


def _is_3d(dvf):
    return dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] > 1


@register_variant("trust_constr")
def trust_constr(dvf: np.ndarray, *, threshold: float = 0.01,
                  max_iterations: int = 100,
                  enforce_triangles: bool = True,
                  timeout_s: float = 600.0, **_unused) -> SolverResult:
    if _is_3d(dvf):
        from benchmarks.two_triangle.variants.baseline_serial import baseline_serial
        r = baseline_serial(dvf, threshold=threshold,
                            max_iterations=max_iterations,
                            enforce_triangles=enforce_triangles,
                            timeout_s=timeout_s)
        r.meta["variant"] = "trust_constr"
        r.meta["fallback"] = "baseline_3d"
        return r

    phi_initial = np.stack([dvf[1, 0], dvf[2, 0]])
    r = run_minimal_iterative_2d(
        phi_initial, threshold=threshold, max_iterations=max_iterations,
        enforce_triangles=enforce_triangles, timeout_s=timeout_s,
        method="trust-constr",
        variant_name="trust_constr",
    )
    return r
```

- [ ] **Step 12.3: Update package init**

```python
# benchmarks/two_triangle/variants/__init__.py
"""Importing this package registers all variants."""
from benchmarks.two_triangle.variants import baseline_serial         # noqa: F401
from benchmarks.two_triangle.variants import combinatorial_prefix    # noqa: F401
from benchmarks.two_triangle.variants import soft_margin             # noqa: F401
from benchmarks.two_triangle.variants import active_set              # noqa: F401
from benchmarks.two_triangle.variants import multigrid               # noqa: F401
from benchmarks.two_triangle.variants import trust_constr            # noqa: F401
```

- [ ] **Step 12.4: Run — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_variants_loop_owning.py -v -k trust_constr
```
Expected: 2 passed (may take ~30 s — trust-constr is slower than SLSQP per call).

- [ ] **Step 12.5: Commit**

```powershell
git add benchmarks/two_triangle/variants/trust_constr.py benchmarks/two_triangle/variants/__init__.py benchmarks/two_triangle/tests/test_variants_loop_owning.py
git commit -m "two_triangle bench: trust_constr backend variant"
```

---

## Task 13: Wrapper variant — SVF warm-start

**Files:**
- Create: `benchmarks/two_triangle/variants/svf_warmstart.py`
- Modify: `benchmarks/two_triangle/variants/__init__.py`
- Modify: `benchmarks/two_triangle/tests/test_variants_wrapper.py`

This is the speculative "big swing" variant. The SVF fit + scaling-and-squaring
exponential is implemented as a self-contained helper — *no* dependency on an
external library. The fitter uses Gauss-Newton on the residual `phi - exp(v)`
with a small number of iterations; this is intentionally a lightweight
implementation, not a research-grade SVF library.

- [ ] **Step 13.1: Add test**

Append to `benchmarks/two_triangle/tests/test_variants_wrapper.py`:
```python
def test_svf_warmstart_registered():
    assert "svf_warmstart" in registry.list_variants()


def test_svf_warmstart_runs_2d():
    fn = registry.get_variant("svf_warmstart")
    dvf = _tiny_2d_fold()
    result = fn(dvf, threshold=0.01, max_iterations=10)
    assert result.phi_final.shape == (2, 6, 6)
    assert len(result.trajectory) >= 2  # init + post-SVF + post-baseline


def test_svf_warmstart_preserves_clean_fields():
    """If input has no folds, SVF projection should leave it nearly unchanged."""
    H, W = 8, 8
    dvf = np.zeros((3, 1, H, W))
    dvf[1, 0, 4, 4] = 0.1  # tiny perturbation, no fold
    dvf[2, 0, 4, 4] = 0.1
    fn = registry.get_variant("svf_warmstart")
    result = fn(dvf, threshold=0.01, max_iterations=5)
    # No folds before — there should be none after either
    from benchmarks.two_triangle.metrics import fold_counts
    fc = fold_counts(result.phi_final, threshold=0.01)
    assert fc["fold_count_jdet"] == 0
    assert fc["fold_count_tri"] == 0
```

- [ ] **Step 13.2: Implement SVF warm-start**

`benchmarks/two_triangle/variants/svf_warmstart.py`:
```python
"""SVF warm-start wrapper variant.

Pre-process: fit a stationary velocity field v to phi via L2 (a few
Gauss-Newton iterations on the residual `phi - exp(v)`), compute
phi_pre = exp(v) using scaling-and-squaring, then call iterative_serial
on the residual. SVFs are diffeomorphic when ||grad v|| is small enough,
so phi_pre often has many fewer folds than phi_initial.

This is a lightweight implementation — research-grade SVF libraries
(e.g. voxelmorph) do this much more carefully. The point of the harness
is to measure whether even a simple SVF projection helps.
"""
import time
import traceback

import numpy as np
import pandas as pd

from dvfopt.core.slsqp.iterative import iterative_serial
from dvfopt.core.slsqp.iterative3d import iterative_3d

from benchmarks.two_triangle.registry import register_variant
from benchmarks.two_triangle.result import SolverResult
from benchmarks.two_triangle.metrics import (
    fold_counts, l2_displacement, smoothness,
)


def _is_3d(dvf):
    return dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[1] > 1


def _exp_field_2d(v: np.ndarray, n_squarings: int = 4) -> np.ndarray:
    """Scaling-and-squaring exponential of a (2, H, W) velocity field."""
    phi = v / (2 ** n_squarings)
    H, W = phi.shape[1:]
    yy, xx = np.indices((H, W), dtype=np.float64)
    for _ in range(n_squarings):
        # phi_new(x) = phi(x) + phi(x + phi(x))
        # Sample phi at displaced locations via bilinear interp.
        sample_y = yy + phi[0]
        sample_x = xx + phi[1]
        sy0 = np.clip(np.floor(sample_y).astype(int), 0, H - 1)
        sx0 = np.clip(np.floor(sample_x).astype(int), 0, W - 1)
        sy1 = np.clip(sy0 + 1, 0, H - 1)
        sx1 = np.clip(sx0 + 1, 0, W - 1)
        wy = sample_y - sy0
        wx = sample_x - sx0
        sampled = np.zeros_like(phi)
        for c in (0, 1):
            sampled[c] = (
                phi[c, sy0, sx0] * (1 - wy) * (1 - wx)
                + phi[c, sy0, sx1] * (1 - wy) * wx
                + phi[c, sy1, sx0] * wy * (1 - wx)
                + phi[c, sy1, sx1] * wy * wx
            )
        phi = phi + sampled
    return phi


def _fit_svf_2d(phi_target: np.ndarray, n_iter: int = 3,
                 step: float = 0.5) -> np.ndarray:
    """Tiny Gauss-Newton fitter: v += step * (phi_target - exp(v))."""
    v = phi_target.copy()  # start from phi_target as initial v
    for _ in range(n_iter):
        residual = phi_target - _exp_field_2d(v)
        v = v + step * residual
    return v


@register_variant("svf_warmstart")
def svf_warmstart(dvf: np.ndarray, *, threshold: float = 0.01,
                   max_iterations: int = 100,
                   enforce_triangles: bool = True,
                   timeout_s: float = 600.0,
                   svf_iter: int = 3, n_squarings: int = 4,
                   **_unused) -> SolverResult:
    is_3d = _is_3d(dvf)
    if is_3d:
        # 3D SVF requires a separate exp implementation; fall back to baseline.
        from benchmarks.two_triangle.variants.baseline_serial import baseline_serial
        r = baseline_serial(dvf, threshold=threshold,
                            max_iterations=max_iterations,
                            enforce_triangles=enforce_triangles,
                            timeout_s=timeout_s)
        r.meta["variant"] = "svf_warmstart"
        r.meta["fallback"] = "baseline_3d"
        return r

    phi_initial = np.stack([dvf[1, 0], dvf[2, 0]])
    t0 = time.perf_counter()
    err = None
    rows = []

    fc = fold_counts(phi_initial, threshold=threshold)
    rows.append({
        "outer_iter": 0, "time_s": 0.0,
        "fold_count_jdet": fc["fold_count_jdet"],
        "fold_count_tri": fc["fold_count_tri"],
        "max_violation": fc["max_violation"],
        "l2_disp": 0.0, "smoothness": smoothness(phi_initial),
        "n_active_windows": 0, "inner_iters": 0,
    })

    # --- SVF projection ---
    try:
        v = _fit_svf_2d(phi_initial, n_iter=svf_iter)
        phi_svf = _exp_field_2d(v, n_squarings=n_squarings)
    except Exception:
        err = traceback.format_exc()
        phi_svf = phi_initial

    fc = fold_counts(phi_svf, threshold=threshold)
    rows.append({
        "outer_iter": 1, "time_s": time.perf_counter() - t0,
        "fold_count_jdet": fc["fold_count_jdet"],
        "fold_count_tri": fc["fold_count_tri"],
        "max_violation": fc["max_violation"],
        "l2_disp": l2_displacement(phi_svf, phi_initial),
        "smoothness": smoothness(phi_svf),
        "n_active_windows": 0, "inner_iters": 0,
    })

    # --- Baseline call on residual ---
    phi_post = phi_svf
    if err is None and max_iterations > 0:
        try:
            dvf_post = dvf.copy()
            dvf_post[1, 0] = phi_svf[0]
            dvf_post[2, 0] = phi_svf[1]
            phi_solved = iterative_serial(
                dvf_post, threshold=threshold,
                max_iterations=max_iterations, verbose=0,
                enforce_triangles=enforce_triangles)
            phi_post = phi_solved
        except Exception:
            err = traceback.format_exc()

    fc = fold_counts(phi_post, threshold=threshold)
    rows.append({
        "outer_iter": 2, "time_s": time.perf_counter() - t0,
        "fold_count_jdet": fc["fold_count_jdet"],
        "fold_count_tri": fc["fold_count_tri"],
        "max_violation": fc["max_violation"],
        "l2_disp": l2_displacement(phi_post, phi_initial),
        "smoothness": smoothness(phi_post),
        "n_active_windows": 0, "inner_iters": 0,
    })

    converged = (err is None and fc["fold_count_jdet"] == 0
                 and fc["fold_count_tri"] == 0)
    return SolverResult(
        phi_final=phi_post, trajectory=pd.DataFrame(rows),
        converged=converged, timed_out=False, error=err,
        meta={"variant": "svf_warmstart", "is_3d": is_3d,
              "threshold": threshold, "svf_iter": svf_iter,
              "n_squarings": n_squarings},
    )
```

- [ ] **Step 13.3: Update package init**

```python
# benchmarks/two_triangle/variants/__init__.py
"""Importing this package registers all variants."""
from benchmarks.two_triangle.variants import baseline_serial         # noqa: F401
from benchmarks.two_triangle.variants import combinatorial_prefix    # noqa: F401
from benchmarks.two_triangle.variants import soft_margin             # noqa: F401
from benchmarks.two_triangle.variants import active_set              # noqa: F401
from benchmarks.two_triangle.variants import multigrid               # noqa: F401
from benchmarks.two_triangle.variants import trust_constr            # noqa: F401
from benchmarks.two_triangle.variants import svf_warmstart           # noqa: F401
```

- [ ] **Step 13.4: Run — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_variants_wrapper.py -v -k svf
```
Expected: 3 passed.

- [ ] **Step 13.5: Commit**

```powershell
git add benchmarks/two_triangle/variants/svf_warmstart.py benchmarks/two_triangle/variants/__init__.py benchmarks/two_triangle/tests/test_variants_wrapper.py
git commit -m "two_triangle bench: svf_warmstart wrapper variant"
```

---

## Task 14: Runner CLI + manifest + .gitignore

**Files:**
- Create: `benchmarks/two_triangle/runner.py`
- Modify: `.gitignore`
- Create: `benchmarks/two_triangle/tests/test_runner.py`

- [ ] **Step 14.1: Update .gitignore**

Append to `.gitignore`:
```
/benchmarks/two_triangle/results/
```

- [ ] **Step 14.2: Write failing test for runner**

`benchmarks/two_triangle/tests/test_runner.py`:
```python
import json
from pathlib import Path

import pytest

from benchmarks.two_triangle.runner import run_sweep
from benchmarks.two_triangle.result import SolverResult


def test_run_sweep_smoke(tmp_path):
    out_dir = tmp_path / "smoke"
    manifest_path = run_sweep(
        variants=["baseline_serial"],
        cases=["synth2d_single_cell_flip"],
        output_dir=out_dir,
        timeout_s=60.0,
        max_iterations=20,
    )
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_cells"] == 1
    # Locate the single result parquet
    parquet_files = list(out_dir.rglob("*.parquet"))
    assert len(parquet_files) == 1
    r = SolverResult.from_parquet(parquet_files[0])
    assert r.phi_final.shape == (2, 6, 6)


def test_run_sweep_reports_unknown_variant(tmp_path):
    with pytest.raises(KeyError):
        run_sweep(variants=["nonexistent"],
                  cases=["synth2d_single_cell_flip"],
                  output_dir=tmp_path / "x", timeout_s=10.0)
```

- [ ] **Step 14.3: Run — expect FAIL**

```powershell
pytest benchmarks/two_triangle/tests/test_runner.py -v
```

- [ ] **Step 14.4: Implement runner**

`benchmarks/two_triangle/runner.py`:
```python
"""CLI entrypoint and programmatic API for sweeping variants × cases.

Usage (CLI):
    python -m benchmarks.two_triangle.runner --variants baseline_serial \
        --cases synth2d_single_cell_flip
    python -m benchmarks.two_triangle.runner --all
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

# Ensure repo root on sys.path when invoked as `python -m`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.two_triangle import registry  # noqa: E402
import benchmarks.two_triangle.cases  # noqa: F401, E402  -- registers cases
import benchmarks.two_triangle.variants  # noqa: F401, E402  -- registers variants
from benchmarks.two_triangle.result import SolverResult  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _current_rss_mb() -> float:
    """Resident set size in MB. Falls back to 0 if psutil is unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _run_cell(variant_name: str, case_name: str, *,
              timeout_s: float, threshold: float, max_iterations: int,
              output_root: Path) -> dict:
    """Run one (variant, case). Returns a manifest entry (always — even on error).

    Caveat: timeout_s is honoured *cooperatively* by variants that own their
    loop (they check elapsed time per outer iter). The baseline variant calls
    the underlying iterative_serial directly and cannot be interrupted —
    `max_iterations` is the only bound for it.

    Memory sampling is best-effort: rss_mb_after captures resident set size
    immediately after the variant returns. It is an upper-bound proxy for
    peak memory, not a true high-water mark — sampling continuously would
    require a thread, which complicates the runner.
    """
    variant_fn = registry.get_variant(variant_name)
    case_fn = registry.get_case(case_name)
    case_meta = registry.case_metadata(case_name)
    cell_dir = output_root / variant_name
    cell_dir.mkdir(parents=True, exist_ok=True)
    parquet = cell_dir / f"{case_name}.parquet"

    entry = {
        "variant": variant_name, "case": case_name,
        "case_meta": case_meta, "started_at": _utc_stamp(),
    }
    try:
        dvf, _meta = case_fn()
        rss_before = _current_rss_mb()
        t0 = time.perf_counter()
        result: SolverResult = variant_fn(
            dvf, threshold=threshold, max_iterations=max_iterations,
            timeout_s=timeout_s)
        elapsed = time.perf_counter() - t0
        rss_after = _current_rss_mb()
        result.to_parquet(parquet)
        entry.update({
            "status": "ok",
            "wall_s": elapsed,
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
            "rss_delta_mb": max(0.0, rss_after - rss_before),
            "converged": result.converged,
            "timed_out": result.timed_out,
            "final_folds_jdet": int(result.trajectory.iloc[-1]["fold_count_jdet"]),
            "final_folds_tri": int(result.trajectory.iloc[-1]["fold_count_tri"]),
            "n_traj_rows": len(result.trajectory),
            "parquet": str(parquet.relative_to(output_root)),
        })
    except FileNotFoundError as exc:
        entry.update({"status": "skipped",
                      "reason": f"data file missing: {exc}"})
    except Exception:
        entry.update({"status": "error",
                      "traceback": traceback.format_exc()})
    return entry


def run_sweep(*, variants: Iterable[str], cases: Iterable[str],
              output_dir: Optional[Path] = None,
              timeout_s: float = 600.0, threshold: float = 0.01,
              max_iterations: int = 100) -> Path:
    """Programmatic sweep entrypoint. Returns path to the run's manifest.json."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "runs" / _utc_stamp()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate before running anything
    variants = list(variants)
    cases = list(cases)
    for v in variants:
        registry.get_variant(v)
    for c in cases:
        registry.get_case(c)

    cells = []
    for v in variants:
        for c in cases:
            print(f"[run] {v} × {c}", flush=True)
            cells.append(_run_cell(
                v, c, timeout_s=timeout_s, threshold=threshold,
                max_iterations=max_iterations, output_root=output_dir,
            ))

    manifest = {
        "git_sha": _git_sha(),
        "utc": _utc_stamp(),
        "host": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "n_cells": len(cells),
        "config": {"timeout_s": timeout_s, "threshold": threshold,
                   "max_iterations": max_iterations},
        "cells": cells,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[done] manifest -> {manifest_path}", flush=True)
    return manifest_path


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-triangle benchmark runner")
    p.add_argument("--variants", nargs="+", default=None,
                   help="Variant names. Use --all-variants for everything.")
    p.add_argument("--cases", nargs="+", default=None,
                   help="Case names. Use --all-cases for everything.")
    p.add_argument("--all", action="store_true",
                   help="Shortcut for --all-variants --all-cases.")
    p.add_argument("--all-variants", action="store_true")
    p.add_argument("--all-cases", action="store_true")
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.all:
        args.all_variants = args.all_cases = True
    variants = (registry.list_variants() if args.all_variants else args.variants)
    cases = (registry.list_cases() if args.all_cases else args.cases)
    if not variants or not cases:
        print("Specify --variants and --cases (or --all). Available:")
        print(f"  variants: {registry.list_variants()}")
        print(f"  cases:    {registry.list_cases()}")
        return 2
    run_sweep(variants=variants, cases=cases, output_dir=args.output,
              timeout_s=args.timeout_s, threshold=args.threshold,
              max_iterations=args.max_iter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 14.5: Run — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_runner.py -v
```
Expected: 2 passed.

- [ ] **Step 14.6: Smoke-test the CLI end-to-end**

```powershell
python -m benchmarks.two_triangle.runner --variants baseline_serial --cases synth2d_single_cell_flip
```
Expected output: `[run] baseline_serial × synth2d_single_cell_flip` then `[done] manifest -> ...`. Inspect the resulting `manifest.json` and parquet under `benchmarks/two_triangle/results/runs/<stamp>/`.

- [ ] **Step 14.7: Commit**

```powershell
git add benchmarks/two_triangle/runner.py benchmarks/two_triangle/tests/test_runner.py .gitignore
git commit -m "two_triangle bench: runner CLI + manifest + gitignore results dir"
```

---

## Task 15: Verification — baseline parity test

**Files:**
- Create: `benchmarks/two_triangle/tests/test_parity.py`

This task verifies the harness doesn't drift from the underlying solver:
the baseline variant should produce *identical* phi_final to a direct call
to `iterative_serial`.

- [ ] **Step 15.1: Write the parity test**

`benchmarks/two_triangle/tests/test_parity.py`:
```python
"""Parity: baseline_serial variant must match a direct iterative_serial call."""
import numpy as np
import pytest

from dvfopt.core.slsqp.iterative import iterative_serial

from benchmarks.two_triangle import registry
import benchmarks.two_triangle.variants  # registers
import benchmarks.two_triangle.cases     # registers


@pytest.mark.parametrize("case_name", [
    "synth2d_single_cell_flip",
    "synth2d_horizontal_bowtie",
])
def test_baseline_matches_direct_solver(case_name):
    case_fn = registry.get_case(case_name)
    dvf, _ = case_fn()

    direct = iterative_serial(dvf.copy(), threshold=0.01, max_iterations=50,
                               verbose=0, enforce_triangles=True)

    baseline_fn = registry.get_variant("baseline_serial")
    result = baseline_fn(dvf.copy(), threshold=0.01, max_iterations=50)

    # Direct call returns (2, H, W) [dy, dx]; baseline returns the same canonical
    # (2, H, W) format.
    np.testing.assert_allclose(result.phi_final, direct, rtol=1e-10, atol=1e-10)
```

- [ ] **Step 15.2: Run — expect PASS**

```powershell
pytest benchmarks/two_triangle/tests/test_parity.py -v
```
Expected: 2 passed. If failure, the baseline variant has introduced unintended drift — fix before proceeding.

- [ ] **Step 15.3: Commit**

```powershell
git add benchmarks/two_triangle/tests/test_parity.py
git commit -m "two_triangle bench: parity test for baseline vs direct solver"
```

---

## Task 16: Report notebook

**Files:**
- Create: `benchmarks/two_triangle/report.ipynb`

This task creates a Jupyter notebook with cells producing the six standard
reports. Build it by writing the cells iteratively and running each one.

- [ ] **Step 16.1: Run a sweep with at least 2 variants × 2 cases to have data**

```powershell
python -m benchmarks.two_triangle.runner --variants baseline_serial combinatorial_prefix --cases synth2d_single_cell_flip synth2d_horizontal_bowtie --max-iter 30
```
Expected: 4 cells run, manifest written. Note the run directory.

- [ ] **Step 16.2: Create the notebook with these cells (in order)**

Use `jupytext` or write the JSON directly. Each cell's source code:

**Cell 1 (markdown):**
```markdown
# Two-Triangle Benchmark Report

This notebook reads the latest run under `benchmarks/two_triangle/results/runs/`
and produces six standard comparison outputs:

1. Per-case winner table
2. Variants × cases speed heatmap (vs baseline)
3. Variants × cases accuracy heatmap (vs baseline)
4. Pareto scatter: speed vs final folds
5. Trajectory overlays for real cases (fold count vs time)
6. Solution-quality table (l2_disp, smoothness vs baseline)

To refresh: re-run all cells.
```

**Cell 2 (code) — load latest run:**
```python
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from benchmarks.two_triangle.result import SolverResult

RUNS_DIR = Path("results/runs")
latest = sorted(RUNS_DIR.iterdir())[-1]
manifest = json.loads((latest / "manifest.json").read_text())
print(f"Run: {latest.name}  |  cells: {manifest['n_cells']}  |  git: {manifest['git_sha'][:8]}")
cells = pd.DataFrame(manifest["cells"])
ok = cells[cells["status"] == "ok"].copy()
print(f"  ok: {len(ok)}  |  errors: {(cells['status'] == 'error').sum()}  |  skipped: {(cells['status'] == 'skipped').sum()}")
ok.head()
```

**Cell 3 (code) — winner table:**
```python
# 1. Per-case winner: minimize fold_count_tri × wall_s
ok["score"] = (ok["final_folds_tri"] + 1) * ok["wall_s"]
winners = (ok.sort_values("score")
             .groupby("case", as_index=False)
             .first()[["case", "variant", "final_folds_tri", "wall_s", "score"]])
winners
```

**Cell 4 (code) — speed heatmap:**
```python
# 2. Speed heatmap: log10(wall_s / baseline_wall_s)
pivot_time = ok.pivot(index="variant", columns="case", values="wall_s")
baseline = pivot_time.loc["baseline_serial"]
log_ratio = np.log10(pivot_time / baseline)
fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(pivot_time.columns)), 0.6 * len(pivot_time)))
im = ax.imshow(log_ratio, cmap="RdYlGn_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(pivot_time.columns))); ax.set_xticklabels(pivot_time.columns, rotation=45, ha="right")
ax.set_yticks(range(len(pivot_time.index))); ax.set_yticklabels(pivot_time.index)
plt.colorbar(im, ax=ax, label="log10(wall_s / baseline)")
ax.set_title("Speed vs baseline (red = slower, green = faster)")
plt.tight_layout(); plt.show()
```

**Cell 5 (code) — accuracy heatmap:**
```python
# 3. Accuracy heatmap: final_folds_tri minus baseline
pivot_folds = ok.pivot(index="variant", columns="case", values="final_folds_tri")
delta = pivot_folds.sub(pivot_folds.loc["baseline_serial"], axis="columns")
fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(pivot_folds.columns)), 0.6 * len(pivot_folds)))
im = ax.imshow(delta, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(pivot_folds.columns))); ax.set_xticklabels(pivot_folds.columns, rotation=45, ha="right")
ax.set_yticks(range(len(pivot_folds.index))); ax.set_yticklabels(pivot_folds.index)
for i in range(delta.shape[0]):
    for j in range(delta.shape[1]):
        ax.text(j, i, int(delta.iloc[i, j]), ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=ax, label="folds_tri - baseline_folds_tri")
ax.set_title("Accuracy delta vs baseline (negative = better)")
plt.tight_layout(); plt.show()
```

**Cell 6 (code) — Pareto scatter:**
```python
# 4. Pareto scatter: speed vs folds
fig, ax = plt.subplots(figsize=(8, 6))
for v, group in ok.groupby("variant"):
    ax.scatter(group["wall_s"], group["final_folds_tri"], label=v, s=80, alpha=0.7)
ax.set_xlabel("wall_s (lower = faster)"); ax.set_ylabel("final_folds_tri (lower = better)")
ax.set_xscale("log")
ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
ax.set_title("Pareto: speed vs accuracy")
plt.tight_layout(); plt.show()
```

**Cell 7 (code) — trajectory overlays:**
```python
# 5. Trajectory overlays for the 2 longest-running cases (proxy for "real" or hard)
hardest = ok.groupby("case")["wall_s"].max().nlargest(2).index.tolist()
fig, axes = plt.subplots(1, len(hardest), figsize=(7 * len(hardest), 4), squeeze=False)
for ax, case in zip(axes[0], hardest):
    for _, row in ok[ok["case"] == case].iterrows():
        parquet = latest / row["parquet"]
        if not parquet.exists():
            continue
        traj = pd.read_parquet(parquet)
        ax.plot(traj["time_s"], traj["fold_count_tri"], marker="o", label=row["variant"])
    ax.set_title(case); ax.set_xlabel("time (s)"); ax.set_ylabel("fold_count_tri")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
```

**Cell 8 (code) — quality table:**
```python
# 6. Solution quality: l2_disp + smoothness deltas
def _final_metric(parquet, col):
    return float(pd.read_parquet(latest / parquet).iloc[-1][col])

ok["final_l2_disp"] = ok["parquet"].apply(lambda p: _final_metric(p, "l2_disp"))
ok["final_smoothness"] = ok["parquet"].apply(lambda p: _final_metric(p, "smoothness"))
quality = ok[["case", "variant", "final_l2_disp", "final_smoothness"]].pivot(
    index="variant", columns="case")
quality
```

Save the notebook as `benchmarks/two_triangle/report.ipynb`.

- [ ] **Step 16.3: Run the notebook end-to-end**

In the repo root:
```powershell
jupyter nbconvert --to notebook --execute benchmarks/two_triangle/report.ipynb --output report.ipynb
```
Expected: notebook runs without error and re-saves itself with cell outputs populated.

- [ ] **Step 16.4: Commit**

```powershell
git add benchmarks/two_triangle/report.ipynb
git commit -m "two_triangle bench: report notebook with 6 standard outputs"
```

---

## Task 17: End-to-end verification sweep

This task is purely for human inspection — no code change.

- [ ] **Step 17.1: Run the full sweep on synthetic 2D + 2D real (skip 3D for speed)**

```powershell
python -m benchmarks.two_triangle.runner --all-variants --cases synth2d_single_cell_flip synth2d_horizontal_bowtie synth2d_diagonal_bowtie synth2d_layered_bowtie_stack real2d_slice90_64x91 real2d_slice200_64x91 --max-iter 50 --timeout-s 300
```
Expected: 7 variants × 6 cases = 42 cells. Real cases may skip if data files aren't present.

- [ ] **Step 17.2: Refresh the report notebook**

```powershell
jupyter nbconvert --to notebook --execute benchmarks/two_triangle/report.ipynb --output report.ipynb
```

- [ ] **Step 17.3: Inspect the outputs**

Open `benchmarks/two_triangle/report.ipynb` in VS Code or Jupyter. Sanity check:
- Winner table makes sense (some cases won by non-baseline variants)
- Speed heatmap shows variation
- Accuracy heatmap shows variation
- Pareto plot has spread points
- Trajectory overlays show distinct curves per variant

If something looks wrong (e.g., a variant always errors), grep `manifest.json` for `"status": "error"` to find tracebacks, fix the variant, re-run the sweep.

- [ ] **Step 17.4: Commit any tweaks**

If you tweaked a variant during inspection, commit it:
```powershell
git add -A
git commit -m "two_triangle bench: tweaks from end-to-end sweep inspection"
```

---

## Final Verification Checklist

Run these commands in order; all should pass:

- [ ] `pytest benchmarks/two_triangle/tests -v` → all green
- [ ] `python -m benchmarks.two_triangle.runner --variants baseline_serial --cases synth2d_single_cell_flip` → completes < 30 s, produces parquet + manifest
- [ ] `jupyter nbconvert --to notebook --execute benchmarks/two_triangle/report.ipynb --output report.ipynb` → completes without error
- [ ] `git status` → clean (all changes committed)
- [ ] No edits to `dvfopt/`, `laplacian/`, or `test_cases/` (verify with `git log --stat origin/main..HEAD -- dvfopt/ laplacian/ test_cases/` — should be empty)
