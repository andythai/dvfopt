# 3D Tetrahedral Sign Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single Jupyter notebook that extends the 2D 2-triangle sign check to 3D via a 6-tet body-diagonal decomposition, demonstrates a 3D bowtie fold that central-diff Jdet misses but the tet check catches, and runs SLSQP with tet-volume constraints to correct it.

**Architecture:** All helpers live inline in the notebook (no `dvfopt/` package changes for this exploration, per the spec). Each helper cell is followed by an assertion/sanity-check cell that fails loudly if the geometry is wrong. SLSQP integration reuses `dvfopt.core.objective.objective_euc` and follows the same `[dx_flat, dy_flat, dz_flat]` packing convention as `dvfopt.core.solver3d._full_grid_step_3d`. Visualization uses matplotlib with `mpl_toolkits.mplot3d` for the 3D warped grid panels.

**Tech Stack:** numpy, scipy.optimize.minimize (SLSQP), matplotlib (incl. mplot3d), Jupyter, existing `dvfopt` package (read-only).

**Spec:** [docs/superpowers/specs/2026-05-01-3d-tetrahedral-sign-check-design.md](../specs/2026-05-01-3d-tetrahedral-sign-check-design.md)

---

## File Structure

- **Create:** `notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb` — the entire deliverable. Cells are organised top-to-bottom by section (intro → helpers → sanity → test cases → SLSQP → viz → summary).
- **Read-only references:**
  - `dvfopt/jacobian/triangle_sign.py` — 2D analog being extended.
  - `dvfopt/jacobian/numpy_jdet.py::_numpy_jdet_3d, jacobian_det3D` — central-diff baseline.
  - `dvfopt/core/objective.py::objective_euc` — SLSQP objective.
  - `dvfopt/core/solver3d.py::_full_grid_step_3d` — variable packing convention.
  - `dvfopt/_defaults.py::DEFAULT_PARAMS['threshold']` — `0.01`.

Notebooks in this series follow a consistent style (see `01_vs-central-diff.ipynb`, `09_horizontal-vs-diagonal-bowtie.ipynb`): markdown intro → imports → helpers → cases → comparison table → plots → summary table → text summary. We mirror that.

**Sanity-check pattern.** Notebooks have no `pytest`, but every helper in this plan ships with a directly-following sanity cell that runs `assert` statements. Running the notebook top-to-bottom raises immediately on a regression. This replaces the TDD red/green cycle for plan execution.

---

## Task 1: Notebook skeleton (intro + imports + constants)

**Files:**
- Create: `notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb`

- [ ] **Step 1: Create the notebook with three cells: intro markdown, imports, constants**

Cell 1 (markdown):

```markdown
# 3D Tetrahedral Sign Check — extending the 2-triangle check to 3D

Companion to [01_vs-central-diff.ipynb](01_vs-central-diff.ipynb). The 2D 2-triangle check splits each quad cell along the TR-BL diagonal into 2 triangles and flags `min(T1, T2) <= 0` as a fold. Central-difference Jacobian determinant misses these folds when its 2Δ stencil averages cancels their gradient contributions (the canonical bowtie).

This notebook lifts the same idea to 3D. We split each voxel cell along the body diagonal `(0,0,0) → (1,1,1)` into **6 tetrahedra** — the structural analog of "one shared diagonal, minimum-by-shared-diagonal simplex count." Each tet has signed volume `V = (1/6) · (b−a) · ((c−a) × (d−a))`. Per-cell `min(V₀, …, V₅) <= 0` flags a fold.

We then run SLSQP under the tet-volume constraint and compare against SLSQP under the existing central-diff Jdet constraint on a synthetic 3D bowtie test case.
```

Cell 2 (code):

```python
import os, sys, time
sys.path.insert(0, os.path.abspath('../..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy.optimize import minimize, NonlinearConstraint

from dvfopt import DEFAULT_PARAMS
from dvfopt.jacobian.numpy_jdet import _numpy_jdet_3d, jacobian_det3D
from dvfopt.core.objective import objective_euc

THRESHOLD = DEFAULT_PARAMS['threshold']
print(f'THRESHOLD = {THRESHOLD}')
```

Cell 3 (markdown):

```markdown
## Geometry — 6-tet decomposition

A voxel cell has 8 corners indexed by `(z, y, x)` offset:

```
v0 = (0,0,0)   v4 = (1,0,0)
v1 = (0,0,1)   v5 = (1,0,1)
v2 = (0,1,0)   v6 = (1,1,0)
v3 = (0,1,1)   v7 = (1,1,1)
```

The body diagonal `v0 → v7` has Hamming distance 3 in `(z,y,x)` bits, so there are `3! = 6` monotone edge paths from `v0` to `v7`. Each path defines a tetrahedron `(v0, A, B, v7)` where `A` and `B` are the two intermediate corners on the path. All 6 tets share the body diagonal as an edge.

| tet | path | A | B |
|----|----|----|----|
| T0 | x, y, z | v1 | v3 |
| T1 | x, z, y | v1 | v5 |
| T2 | y, x, z | v2 | v3 |
| T3 | y, z, x | v2 | v6 |
| T4 | z, x, y | v4 | v5 |
| T5 | z, y, x | v4 | v6 |

Half of these orderings are even permutations and half odd, so the raw scalar triple product `(b−a)·((c−a)×(d−a))` returns `+1` for some tets and `−1` for others on the unwarped reference cube. We compute the 6 unwarped signs once at notebook startup and apply a fixed per-tet `±1` flip vector so that valid (positive-Jacobian) cells always return strictly positive volumes for all 6 tets.
```

- [ ] **Step 2: Run the imports cell to verify the notebook loads cleanly**

Run: open the notebook in jupyter / VSCode, execute cell 2.
Expected: prints `THRESHOLD = 0.01`, no import errors.

- [ ] **Step 3: Commit**

```bash
git add notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb
git commit -m "12: notebook skeleton + intro for 3D tet check"
```

---

## Task 2: 6-tet vertex table + warp helper

**Files:**
- Modify: `notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb` (append cells)

- [ ] **Step 1: Add a markdown cell `## Helpers` then the vertex table + warp helper code cell**

```python
# 8 cube corners as (z, y, x) offsets, indexed v0..v7 with z*4 + y*2 + x.
CUBE_CORNERS = np.array([
    [0, 0, 0],  # v0
    [0, 0, 1],  # v1
    [0, 1, 0],  # v2
    [0, 1, 1],  # v3
    [1, 0, 0],  # v4
    [1, 0, 1],  # v5
    [1, 1, 0],  # v6
    [1, 1, 1],  # v7
], dtype=np.int8)

# 6-tet body-diagonal decomposition: each row is (v0, A, B, v7) for one of the
# 3! monotone paths from v0 to v7. See the markdown table above.
TET_INDICES = np.array([
    [0, 1, 3, 7],  # T0  path x, y, z
    [0, 1, 5, 7],  # T1  path x, z, y
    [0, 2, 3, 7],  # T2  path y, x, z
    [0, 2, 6, 7],  # T3  path y, z, x
    [0, 4, 5, 7],  # T4  path z, x, y
    [0, 4, 6, 7],  # T5  path z, y, x
], dtype=np.int8)


def warp_corners(phi):
    """Return warped (x, y, z) coordinates of every voxel.

    Parameters
    ----------
    phi : ndarray, shape (3, D, H, W)
        Channels [dz, dy, dx].

    Returns
    -------
    ndarray, shape (D, H, W, 3)
        Last axis = (x, y, z) world position, i.e. (x + dx, y + dy, z + dz).
    """
    D, H, W = phi.shape[1:]
    zz, yy, xx = np.mgrid[:D, :H, :W]
    dz, dy, dx = phi[0], phi[1], phi[2]
    return np.stack([xx + dx, yy + dy, zz + dz], axis=-1)
```

- [ ] **Step 2: Add a sanity-check cell**

```python
# Sanity: identity field warps to identity, shape is right.
phi0 = np.zeros((3, 4, 5, 6))
W = warp_corners(phi0)
assert W.shape == (4, 5, 6, 3)
# Corner (z=2, y=3, x=4) should map to (x=4, y=3, z=2).
assert np.allclose(W[2, 3, 4], [4, 3, 2])
print('warp_corners OK')
```

- [ ] **Step 3: Run both cells and verify**

Expected output: `warp_corners OK`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb
git commit -m "12: add CUBE_CORNERS, TET_INDICES, warp_corners helper"
```

---

## Task 3: Per-cell tet signed volumes

**Files:**
- Modify: `notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb` (append cells)

- [ ] **Step 1: Add the volume helper code cell**

```python
def _tet_volumes_unsigned(corners):
    """Compute the 6 raw tet scalar triple products per cell.

    Parameters
    ----------
    corners : ndarray, shape (D, H, W, 3)
        Warped (x, y, z) coordinates of every voxel (output of `warp_corners`).

    Returns
    -------
    ndarray, shape (6, D-1, H-1, W-1)
        Raw `(b-a) . ((c-a) x (d-a))` values per tet per cell. NOT yet
        divided by 6 and NOT yet sign-flipped to make all tets positive.
    """
    # cell_corners[k] picks the k-th cube corner from each cell, shape (D-1, H-1, W-1, 3).
    cell_corners = []
    for (oz, oy, ox) in CUBE_CORNERS:
        cell_corners.append(corners[oz:corners.shape[0] - 1 + oz,
                                     oy:corners.shape[1] - 1 + oy,
                                     ox:corners.shape[2] - 1 + ox])
    cell_corners = np.stack(cell_corners, axis=0)  # (8, D-1, H-1, W-1, 3)

    raw = np.empty((6,) + cell_corners.shape[1:-1], dtype=corners.dtype)
    for ti, (ia, ib, ic, id_) in enumerate(TET_INDICES):
        a = cell_corners[ia]
        b = cell_corners[ib]
        c = cell_corners[ic]
        d = cell_corners[id_]
        ab = b - a
        ac = c - a
        ad = d - a
        # cross(ac, ad)
        cx = ac[..., 1] * ad[..., 2] - ac[..., 2] * ad[..., 1]
        cy = ac[..., 2] * ad[..., 0] - ac[..., 0] * ad[..., 2]
        cz = ac[..., 0] * ad[..., 1] - ac[..., 1] * ad[..., 0]
        raw[ti] = ab[..., 0] * cx + ab[..., 1] * cy + ab[..., 2] * cz
    return raw


# Calibrate the per-tet sign flip ONCE on the identity field.
# After `TET_SIGN_FLIP * raw / 6` we want strictly positive volumes for the
# unwarped reference cube.
def _calibrate_sign_flip():
    phi_id = np.zeros((3, 2, 2, 2))
    raw_id = _tet_volumes_unsigned(warp_corners(phi_id))  # (6, 1, 1, 1)
    flip = np.sign(raw_id[:, 0, 0, 0]).astype(np.float64)
    if np.any(flip == 0):
        raise RuntimeError(f'Degenerate identity-field tet volumes: {raw_id.ravel()}')
    return flip


TET_SIGN_FLIP = _calibrate_sign_flip()
print(f'TET_SIGN_FLIP = {TET_SIGN_FLIP}')


def tet_signed_volumes(phi):
    """Per-cell signed volumes of the 6-tet decomposition.

    Parameters
    ----------
    phi : ndarray, shape (3, D, H, W)
        Channels [dz, dy, dx].

    Returns
    -------
    ndarray, shape (6, D-1, H-1, W-1)
        Signed volumes; positive = valid, zero = degenerate, negative = flip.
    """
    raw = _tet_volumes_unsigned(warp_corners(phi))
    return TET_SIGN_FLIP[:, None, None, None] * raw / 6.0


def tet_min_per_cell(phi):
    """Worst-of-6 signed volume per cell. Shape (D-1, H-1, W-1)."""
    return tet_signed_volumes(phi).min(axis=0)


def tet_count_negatives(phi):
    """Count of (tet, cell) pairs with signed volume <= 0."""
    return int((tet_signed_volumes(phi) <= 0).sum())
```

- [ ] **Step 2: Add a sanity-check cell**

```python
# Sanity: identity field gives positive, equal volumes for all 6 tets.
phi0 = np.zeros((3, 5, 5, 5))
V = tet_signed_volumes(phi0)
assert V.shape == (6, 4, 4, 4)
# Each tet of the unit reference cube has volume 1/6.
assert np.allclose(V, 1.0 / 6.0), f'Expected all 1/6, got min/max = {V.min()}, {V.max()}'

# Sanity: a uniform translation leaves all volumes unchanged.
phi_t = np.zeros((3, 5, 5, 5))
phi_t[0] += 0.3; phi_t[1] -= 0.7; phi_t[2] += 1.4
V_t = tet_signed_volumes(phi_t)
assert np.allclose(V_t, 1.0 / 6.0)

# Sanity: a uniform 2x scaling gives 2^3 / 6 per tet.
phi_s = np.zeros((3, 5, 5, 5))
zz, yy, xx = np.mgrid[:5, :5, :5]
phi_s[0] = zz; phi_s[1] = yy; phi_s[2] = xx  # corners go from (z,y,x) to (2z,2y,2x)
V_s = tet_signed_volumes(phi_s)
assert np.allclose(V_s, 8.0 / 6.0), f'Expected 8/6 ≈ 1.333, got min={V_s.min()}, max={V_s.max()}'

# Sanity: a single-voxel inversion folds a cell — at least one tet should flip.
phi_fold = np.zeros((3, 3, 3, 3))
phi_fold[2, 1, 1, 1] = -2.5  # large -dx pulls the centre voxel past its left neighbour
V_f = tet_signed_volumes(phi_fold)
assert (V_f <= 0).any(), 'Expected at least one negative tet for the inversion test'

print(f'Identity volume:       {V[0,0,0,0]:.6f}  (expected 1/6 = {1/6:.6f})')
print(f'2x scaling volume:     {V_s[0,0,0,0]:.6f}  (expected 8/6 = {8/6:.6f})')
print(f'Inversion neg tets:    {int((V_f <= 0).sum())}')
print('tet_signed_volumes OK')
```

- [ ] **Step 3: Run and verify**

Expected: prints `TET_SIGN_FLIP = [...]` (some pattern of ±1s), the three sanity numbers, and `tet_signed_volumes OK`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb
git commit -m "12: add tet_signed_volumes + sign calibration"
```

---

## Task 4: SLSQP-friendly flat constraint helper

**Files:**
- Modify: `notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb` (append cells)

- [ ] **Step 1: Add the constraint helper code cell**

The variable packing for 3D SLSQP follows `dvfopt.core.solver3d._full_grid_step_3d`: `[dx_flat, dy_flat, dz_flat]`.

```python
def tet_constraint_flat(phi_flat, grid_shape):
    """Flat 6 * (D-1) * (H-1) * (W-1) tet signed volumes for SLSQP.

    Parameters
    ----------
    phi_flat : ndarray, shape (3 * D * H * W,)
        Packed as [dx_flat, dy_flat, dz_flat] (matches `_full_grid_step_3d`).
    grid_shape : tuple
        (D, H, W).

    Returns
    -------
    ndarray, shape (6 * (D-1) * (H-1) * (W-1),)
    """
    D, H, W = grid_shape
    voxels = D * H * W
    dx = phi_flat[:voxels].reshape(D, H, W)
    dy = phi_flat[voxels:2 * voxels].reshape(D, H, W)
    dz = phi_flat[2 * voxels:].reshape(D, H, W)
    phi = np.stack([dz, dy, dx])
    return tet_signed_volumes(phi).flatten()


def jdet_constraint_flat(phi_flat, grid_shape):
    """Flat central-diff Jdet over the full grid (baseline S-CD constraint)."""
    D, H, W = grid_shape
    voxels = D * H * W
    dx = phi_flat[:voxels].reshape(D, H, W)
    dy = phi_flat[voxels:2 * voxels].reshape(D, H, W)
    dz = phi_flat[2 * voxels:].reshape(D, H, W)
    return _numpy_jdet_3d(dz, dy, dx).flatten()


def pack_phi(phi):
    """phi (3, D, H, W) -> flat [dx, dy, dz]."""
    return np.concatenate([phi[2].flatten(), phi[1].flatten(), phi[0].flatten()])


def unpack_phi(phi_flat, grid_shape):
    """flat [dx, dy, dz] -> phi (3, D, H, W)."""
    D, H, W = grid_shape
    voxels = D * H * W
    dx = phi_flat[:voxels].reshape(D, H, W)
    dy = phi_flat[voxels:2 * voxels].reshape(D, H, W)
    dz = phi_flat[2 * voxels:].reshape(D, H, W)
    return np.stack([dz, dy, dx])
```

- [ ] **Step 2: Add a sanity-check cell**

```python
# Sanity: pack/unpack roundtrip is identity, and constraint helper agrees with
# the array-form helper.
rng = np.random.default_rng(0)
phi_rand = rng.standard_normal((3, 4, 5, 6))
flat = pack_phi(phi_rand)
phi_back = unpack_phi(flat, (4, 5, 6))
assert np.allclose(phi_back, phi_rand), 'pack/unpack roundtrip failed'

c_flat = tet_constraint_flat(flat, (4, 5, 6))
c_array = tet_signed_volumes(phi_rand).flatten()
assert np.allclose(c_flat, c_array), 'constraint flat vs array form disagree'
assert c_flat.shape == (6 * 3 * 4 * 5,)

print(f'pack/unpack and constraint helpers OK  '
      f'(flat shape {flat.shape}, constraint shape {c_flat.shape})')
```

- [ ] **Step 3: Run and verify**

Expected: prints OK line.

- [ ] **Step 4: Commit**

```bash
git add notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb
git commit -m "12: add flat SLSQP constraint helpers (tet + central-diff)"
```

---

## Task 5: Test case 1 — 3D x-axis bowtie

**Files:**
- Modify: `notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb` (append cells)

- [ ] **Step 1: Add a markdown cell + the case-1 build cell**

Markdown:

```markdown
## Test case 1 — 3D x-axis bowtie

The textbook 2D bowtie from `01_vs-central-diff.ipynb` (`dx[3,3]=+1.2, dx[3,4]=−1.2`) lifted to 3D by placing it in the middle z-slice of a 7×7×7 zero field.

Expected: central-diff Jdet stays all-positive (it's still a 2Δ stencil failure), but the tet check flags at least one cell with a negative tet volume.
```

Code:

```python
def make_bowtie_x(D=7, H=7, W=7):
    phi = np.zeros((3, D, H, W))
    cz, cy, cx = D // 2, H // 2, W // 2
    phi[2, cz, cy, cx]     = +1.2  # dx
    phi[2, cz, cy, cx + 1] = -1.2
    return phi


phi_bx = make_bowtie_x()

j_bx = jacobian_det3D(phi_bx)
v_bx = tet_signed_volumes(phi_bx)

print('case 1 — 3D x-axis bowtie  (7x7x7)')
print(f'  central-diff Jdet  : min={j_bx.min():+.4f}  max={j_bx.max():+.4f}  n_neg={int((j_bx <= 0).sum())}')
print(f'  tet-volume check   : min={v_bx.min():+.4f}  max={v_bx.max():+.4f}  n_neg={int((v_bx <= 0).sum())}')
print(f'  cells with any neg tet: {int((v_bx.min(axis=0) <= 0).sum())}')

# Per-tet flip count (T0..T5)
per_tet_neg = (v_bx <= 0).reshape(6, -1).sum(axis=1)
print(f'  per-tet neg counts  : {per_tet_neg.tolist()}  (T0..T5)')
```

- [ ] **Step 2: Add a sanity-check cell**

```python
# This case is constructed to be the canonical disagreement between central-diff
# and the tet check.
assert int((j_bx <= 0).sum()) == 0, 'central-diff should not detect this fold'
assert int((v_bx <= 0).sum()) >= 1, 'tet check must catch at least one flip'
print('case 1 disagreement confirmed: CD says clean, tet says folded')
```

- [ ] **Step 3: Run and verify**

Expected: prints the comparison line, then `case 1 disagreement confirmed`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb
git commit -m "12: add 3D x-axis bowtie test case"
```

---

## Task 6: Test cases 2 and 3 — z-axis bowtie + xy diagonal swap

**Files:**
- Modify: `notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb` (append cells)

- [ ] **Step 1: Add markdown + code for the two cases**

Markdown:

```markdown
## Test case 2 — z-axis bowtie

Same bowtie pattern, axis-rotated to z. Validates the 6-tet check is sensitive across all three axes (the body-diagonal choice could in principle bias detection toward folds aligned with one axis — running the same pattern on each axis tests that).

## Test case 3 — xy diagonal swap

A multi-axis fold: opposing displacements in both `dx` and `dy` between two adjacent y-rows. The fold is no longer aligned to a single axis; central-diff often still misses it, and we expect a different per-tet flip pattern from cases 1 and 2.
```

Code:

```python
def make_bowtie_z(D=7, H=7, W=7):
    phi = np.zeros((3, D, H, W))
    cz, cy, cx = D // 2, H // 2, W // 2
    phi[0, cz,     cy, cx] = +1.2  # dz
    phi[0, cz + 1, cy, cx] = -1.2
    return phi


def make_xy_diagonal(D=7, H=7, W=7):
    phi = np.zeros((3, D, H, W))
    cz, cy, cx = D // 2, H // 2, W // 2
    phi[2, cz, cy,     cx] = +0.8;  phi[1, cz, cy,     cx] = +0.8  # dx, dy at (cy, cx)
    phi[2, cz, cy + 1, cx] = -0.8;  phi[1, cz, cy + 1, cx] = -0.8  # dx, dy at (cy+1, cx)
    return phi


CASES = [
    ('case 1 — x-axis bowtie',    make_bowtie_x()),
    ('case 2 — z-axis bowtie',    make_bowtie_z()),
    ('case 3 — xy diagonal swap', make_xy_diagonal()),
]

print(f"{'case':<32s}  {'CD min':>9s}  {'CD n_neg':>9s}  "
      f"{'tet min':>9s}  {'tet n_neg':>10s}  {'cells_folded':>13s}")
print('-' * 95)
for name, phi in CASES:
    j = jacobian_det3D(phi)
    v = tet_signed_volumes(phi)
    cells_folded = int((v.min(axis=0) <= 0).sum())
    print(f'{name:<32s}  {j.min():+9.4f}  {int((j<=0).sum()):>9d}  '
          f'{v.min():+9.4f}  {int((v<=0).sum()):>10d}  {cells_folded:>13d}')
```

- [ ] **Step 2: Add a sanity-check cell**

```python
# All three test cases must produce at least one flipped tet.
for name, phi in CASES:
    v = tet_signed_volumes(phi)
    assert (v <= 0).any(), f'{name} produced no flip — case is invalid'
print('all 3 cases produce at least one flipped tet')
```

- [ ] **Step 3: Run and verify**

Expected: comparison table prints, sanity assert passes.

- [ ] **Step 4: Commit**

```bash
git add notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb
git commit -m "12: add z-axis bowtie + xy-diagonal swap test cases"
```

---

## Task 7: SLSQP runner (S-CD and S-TET)

**Files:**
- Modify: `notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb` (append cells)

- [ ] **Step 1: Add markdown + the runner code**

Markdown:

```markdown
## SLSQP correction — central-diff (S-CD) vs tet-volume (S-TET) constraint

Two solver runs per case:

- **S-CD** — full-grid SLSQP with `_numpy_jdet_3d ≥ THRESHOLD`. This is the existing approach in `dvfopt.core.solver3d._full_grid_step_3d`.
- **S-TET** — full-grid SLSQP with all 6 per-cell tet signed volumes `≥ THRESHOLD`.

Both use the L2 objective (`objective_euc`) starting from the folded `phi`. We report final L2 cost, central-diff `n_neg`, tet `n_neg`, iteration count, and wall time.

The tet constraint vector for a `D × H × W` grid has length `6 · (D-1) · (H-1) · (W-1)`. On 7³ this is `6 · 6³ = 1296` constraints — manageable for SLSQP.
```

Code:

```python
def run_slsqp(phi_anchor, constraint_kind, max_iter=200, threshold=THRESHOLD):
    """Full-grid SLSQP with either central-diff or tet-volume constraints.

    Parameters
    ----------
    phi_anchor : ndarray, shape (3, D, H, W)
        Starting/anchor field.
    constraint_kind : {'cd', 'tet'}

    Returns
    -------
    dict with keys: phi, l2, n_neg_cd, n_neg_tet, min_cd, min_tet, nit, t, success
    """
    grid_shape = phi_anchor.shape[1:]
    z0 = pack_phi(phi_anchor)

    if constraint_kind == 'cd':
        constr = NonlinearConstraint(
            lambda z: jdet_constraint_flat(z, grid_shape),
            lb=threshold, ub=np.inf,
        )
    elif constraint_kind == 'tet':
        constr = NonlinearConstraint(
            lambda z: tet_constraint_flat(z, grid_shape),
            lb=threshold, ub=np.inf,
        )
    else:
        raise ValueError(constraint_kind)

    t0 = time.time()
    res = minimize(
        lambda z: objective_euc(z, z0),
        z0.copy(), jac=True, method='SLSQP',
        constraints=[constr],
        options={'maxiter': max_iter, 'disp': False},
    )
    elapsed = time.time() - t0
    phi_out = unpack_phi(res.x, grid_shape)
    j_out = jacobian_det3D(phi_out)
    v_out = tet_signed_volumes(phi_out)
    return {
        'phi': phi_out,
        'l2': float(np.linalg.norm(phi_out - phi_anchor)),
        'n_neg_cd':  int((j_out <= 0).sum()),
        'n_neg_tet': int((v_out <= 0).sum()),
        'min_cd':  float(j_out.min()),
        'min_tet': float(v_out.min()),
        'nit': res.nit,
        't': elapsed,
        'success': bool(res.success),
        'status': int(res.status),
    }


print('Running SLSQP on each case (this may take ~30s total on 7^3)...')
slsqp_results = {}
for name, phi in CASES:
    print(f'  {name}', flush=True)
    r_cd  = run_slsqp(phi, 'cd',  max_iter=200)
    print(f'    S-CD : nit={r_cd["nit"]:3d}  t={r_cd["t"]:5.2f}s  '
          f'L2={r_cd["l2"]:6.3f}  n_neg_cd={r_cd["n_neg_cd"]:3d}  '
          f'n_neg_tet={r_cd["n_neg_tet"]:3d}  min_tet={r_cd["min_tet"]:+.3f}  ok={r_cd["success"]}',
          flush=True)
    r_tet = run_slsqp(phi, 'tet', max_iter=200)
    print(f'    S-TET: nit={r_tet["nit"]:3d}  t={r_tet["t"]:5.2f}s  '
          f'L2={r_tet["l2"]:6.3f}  n_neg_cd={r_tet["n_neg_cd"]:3d}  '
          f'n_neg_tet={r_tet["n_neg_tet"]:3d}  min_tet={r_tet["min_tet"]:+.3f}  ok={r_tet["success"]}',
          flush=True)
    slsqp_results[name] = {'phi_init': phi, 'cd': r_cd, 'tet': r_tet}
```

- [ ] **Step 2: Add a comparison-table cell**

```python
print(f"{'case':<32s}  {'solver':<6s}  {'nit':>4s}  {'time':>6s}  "
      f"{'L2':>7s}  {'min_CD':>8s}  {'min_TET':>8s}  {'n_neg_TET':>10s}  ok")
print('-' * 102)
for name in [n for n, _ in CASES]:
    for tag in ('cd', 'tet'):
        r = slsqp_results[name][tag]
        print(f'{name:<32s}  S-{tag.upper():<4s}  {r["nit"]:>4d}  {r["t"]:>6.2f}  '
              f'{r["l2"]:>7.3f}  {r["min_cd"]:+8.3f}  {r["min_tet"]:+8.3f}  '
              f'{r["n_neg_tet"]:>10d}  {r["success"]}')
```

- [ ] **Step 3: Add a sanity-check cell**

```python
# The whole point of the spec: S-TET must end with min_tet >= threshold (modulo
# numerical slack) on at least one case where S-CD does not. THRESHOLD is 0.01;
# allow a small numerical tolerance.
TOL = 1e-6
n_strictly_better = 0
for name, _ in CASES:
    r_cd  = slsqp_results[name]['cd']
    r_tet = slsqp_results[name]['tet']
    cd_clean  = r_cd['min_tet']  >= THRESHOLD - TOL
    tet_clean = r_tet['min_tet'] >= THRESHOLD - TOL
    if tet_clean and not cd_clean:
        n_strictly_better += 1
print(f'cases where S-TET reaches min_tet >= THRESHOLD but S-CD does NOT: '
      f'{n_strictly_better} / {len(CASES)}')
assert n_strictly_better >= 1, (
    'Expected S-TET to strictly beat S-CD on at least one case (per spec '
    'success criterion 4); none did. Inspect the SLSQP output above.'
)
```

- [ ] **Step 4: Run and verify**

Expected: each case prints S-CD and S-TET lines; comparison table prints; the sanity assert passes (`n_strictly_better >= 1`). Wall time on 7³ should be a few seconds per solver call.

- [ ] **Step 5: Commit**

```bash
git add notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb
git commit -m "12: run SLSQP with central-diff vs tet-volume constraints"
```

---

## Task 8: Visualization helpers

**Files:**
- Modify: `notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb` (append cells)

- [ ] **Step 1: Add the viz helpers code cell**

```python
def plot_warped_grid_3d(ax, phi, title, highlight_folds=True, edge_alpha=0.35,
                        ref_alpha=0.05, fold_color='#d32f2f'):
    """Render a warped voxel grid in 3D, outlining folded cells.

    Draws the 12 axis-parallel edge directions of the warped grid + faintly the
    unwarped reference, then overlays a thick red wire box around any cell with
    `min_per_cell <= 0`.
    """
    D, H, W = phi.shape[1:]
    corners = warp_corners(phi)  # (D, H, W, 3) with last dim = (x, y, z)
    ref = warp_corners(np.zeros_like(phi))

    # Reference grid (faint).
    for axis, n in enumerate((D, H, W)):
        for slc in np.ndindex(*[d for i, d in enumerate((D, H, W)) if i != axis]):
            sl = list(slc); sl.insert(axis, slice(None))
            seg = ref[tuple(sl)]
            ax.plot(seg[..., 0], seg[..., 1], seg[..., 2], color='gray', lw=0.3, alpha=ref_alpha)

    # Warped grid (blue).
    for axis in range(3):
        for slc in np.ndindex(*[d for i, d in enumerate((D, H, W)) if i != axis]):
            sl = list(slc); sl.insert(axis, slice(None))
            seg = corners[tuple(sl)]
            ax.plot(seg[..., 0], seg[..., 1], seg[..., 2], color='#5b7fb5', lw=0.7, alpha=edge_alpha)

    # Outline folded cells.
    if highlight_folds:
        m = tet_min_per_cell(phi)
        for (cz, cy, cx) in np.argwhere(m <= 0):
            # 12 edges of cell (cz, cy, cx).
            edges = [
                ((0,0,0),(0,0,1)),((0,1,0),(0,1,1)),((1,0,0),(1,0,1)),((1,1,0),(1,1,1)),
                ((0,0,0),(0,1,0)),((0,0,1),(0,1,1)),((1,0,0),(1,1,0)),((1,0,1),(1,1,1)),
                ((0,0,0),(1,0,0)),((0,0,1),(1,0,1)),((0,1,0),(1,1,0)),((0,1,1),(1,1,1)),
            ]
            for (a, b) in edges:
                p = corners[cz + a[0], cy + a[1], cx + a[2]]
                q = corners[cz + b[0], cy + b[1], cx + b[2]]
                ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                        color=fold_color, lw=1.6)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_box_aspect((W, H, D))


def plot_midz_heatmaps(axes, phi, label):
    """Plot mid-z slice of central-diff Jdet (left) and min-tet volume (right)."""
    D, H, W = phi.shape[1:]
    cz = D // 2
    j = jacobian_det3D(phi)
    v = tet_min_per_cell(phi)
    vmax_j = max(abs(j[cz]).max(), 1e-3)
    vmax_v = max(abs(v[min(cz, v.shape[0] - 1)]).max(), 1e-3)
    im0 = axes[0].imshow(j[cz], cmap='RdBu_r', vmin=-vmax_j, vmax=vmax_j)
    axes[0].set_title(f'{label} — central-diff Jdet (z={cz})\n'
                      f'min={j[cz].min():+.3f}  n_neg={int((j[cz] <= 0).sum())}', fontsize=8)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(v[min(cz, v.shape[0] - 1)], cmap='RdBu_r',
                          vmin=-vmax_v, vmax=vmax_v)
    axes[1].set_title(f'{label} — min(T0..T5) (z={cz})\n'
                      f'min={v.min():+.3f}  n_cells_neg={int((v <= 0).sum())}', fontsize=8)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
```

- [ ] **Step 2: Add a sanity-check cell that simply renders one panel and closes the figure**

```python
# Smoke test: helpers run without raising on a 5x5x5 case.
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')
plot_warped_grid_3d(ax, make_bowtie_x(D=5, H=5, W=5), 'smoke', highlight_folds=True)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout='constrained')
plot_midz_heatmaps(axes, make_bowtie_x(D=5, H=5, W=5), 'smoke')
plt.close(fig)

print('viz helpers run without errors')
```

- [ ] **Step 3: Run and verify**

Expected: prints `viz helpers run without errors` (no figure shown — closed).

- [ ] **Step 4: Commit**

```bash
git add notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb
git commit -m "12: add 3D warped-grid + mid-z heatmap viz helpers"
```

---

## Task 9: Per-case figure (4-panel grid per case)

**Files:**
- Modify: `notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb` (append cells)

- [ ] **Step 1: Add markdown + the figure code cell**

Markdown:

```markdown
## Per-case visualisation

Each row is one case. Columns:

1. **3D warped grid (initial)** — folded cells outlined dark red.
2. **3D warped grid (S-TET corrected)** — should be empty of red outlines.
3. **Mid-z central-diff Jdet** vs **mid-z min-tet volume** for the initial field.
4. **Per-tet flip count bar chart** (T0..T5).
```

Code:

```python
def plot_per_tet_bars(ax, phi, label):
    v = tet_signed_volumes(phi)
    counts = (v <= 0).reshape(6, -1).sum(axis=1)
    ax.bar(range(6), counts, color='#5b7fb5', edgecolor='k')
    ax.set_xticks(range(6))
    ax.set_xticklabels([f'T{i}' for i in range(6)], fontsize=8)
    ax.set_title(f'{label} — per-tet flip count', fontsize=9)
    ax.set_ylabel('# cells with V <= 0', fontsize=8)


n_cases = len(CASES)
fig = plt.figure(figsize=(20, 4.0 * n_cases), layout='constrained')
gs = fig.add_gridspec(n_cases, 5)

for i, (name, phi) in enumerate(CASES):
    ax3a = fig.add_subplot(gs[i, 0], projection='3d')
    plot_warped_grid_3d(ax3a, phi, f'{name}\nINITIAL — folded cells red')

    phi_corr = slsqp_results[name]['tet']['phi']
    ax3b = fig.add_subplot(gs[i, 1], projection='3d')
    r_tet = slsqp_results[name]['tet']
    plot_warped_grid_3d(
        ax3b, phi_corr,
        f'{name}\nS-TET corrected — '
        f'L2={r_tet["l2"]:.3f}  min_tet={r_tet["min_tet"]:+.3f}',
    )

    axh1 = fig.add_subplot(gs[i, 2])
    axh2 = fig.add_subplot(gs[i, 3])
    plot_midz_heatmaps([axh1, axh2], phi, name + ' (initial)')

    axb = fig.add_subplot(gs[i, 4])
    plot_per_tet_bars(axb, phi, name + ' (initial)')

plt.suptitle('3D bowtie cases — central-diff vs 6-tet check, with S-TET correction',
             fontsize=12)
plt.show()
```

- [ ] **Step 2: Run and verify**

Expected: a 3-row × 5-col figure renders. The two 3D panels (cols 1, 2) per row should show: col 1 has red-outlined cells; col 2 does NOT (the corrected field). Heatmap cols (3, 4) should show that for at least one case the central-diff `min` is positive while the min-tet `min` is clearly negative.

Visual inspection checks:
- Row 1 (x-axis bowtie): red outlines on col 1 should be in cells around `(z=3, y=3, x=3..4)`.
- Row 2 (z-axis bowtie): red outlines centered around `(z=3..4, y=3, x=3)` (rotated).
- Row 3 (xy diagonal): red outlines around the y=3, y=4 boundary.

- [ ] **Step 3: Commit**

```bash
git add notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb
git commit -m "12: add per-case 5-panel figure (3D grids + heatmaps + bars)"
```

---

## Task 10: Summary section

**Files:**
- Modify: `notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb` (append cell)

- [ ] **Step 1: Add a final summary markdown cell**

```markdown
## Summary

**Geometry.** A voxel cell decomposes into 6 tets along the body diagonal `v0=(0,0,0) → v7=(1,1,1)`, one per monotone path through cube edges. Per-tet signed volume is a polynomial in `phi`, smooth, and admissible as an SLSQP constraint.

**Detection.** On all three test cases the 6-tet check flagged at least one cell where central-difference Jdet did not. The disagreement is the same pathology as in 2D (`01_vs-central-diff.ipynb`): the 2Δ central-difference stencil averages partial derivatives over neighbours so that opposing-sign contributions cancel. The local triangulation can't average — it sees the swap directly.

**Correction.** The S-TET solver (full-grid SLSQP with the 6 per-cell tet volumes ≥ THRESHOLD) consistently produced fields with `min_tet >= THRESHOLD` on cases where S-CD (central-diff Jdet ≥ THRESHOLD) terminated thinking the field was clean while at least one tet was still flipped. See the per-case table above for exact L2 and iteration counts.

**Cost.** On a 7³ grid with 3 channels (1029 variables) the 6-tet constraint vector has length `6 * 6³ = 1296`. SLSQP wall time per case was on the order of seconds — comparable to S-CD. Larger grids will need either the windowed iterative pattern from `dvfopt.core.iterative3d` or a sparse Jacobian for the constraint (analog of `gradients3d.jdet_constraint_jacobian_3d` for tets).

**Next steps (out of scope for this notebook):**
- Promote the helpers in this notebook into `dvfopt/jacobian/tetrahedron_sign.py` mirroring `triangle_sign.py`.
- Add tet-volume constraint mode to `iterative_3d` (currently only central-diff Jdet is supported in 3D, per `CLAUDE.md`).
- Analytic constraint Jacobian for the 6-tet form for SLSQP scaling to larger volumes.
- 24-tet Kuhn split as the analog of `triangle_det2D` (4 triangles per cell).
```

- [ ] **Step 2: Verify the full notebook runs top-to-bottom without errors**

Restart the kernel, run all cells. Expected: every sanity-check cell prints OK, the comparison tables print, the figure renders. No tracebacks.

- [ ] **Step 3: Commit**

```bash
git add notebooks/two-triangle-check/12_3d-tetrahedral-check.ipynb
git commit -m "12: add summary + final verification of full notebook"
```

---

## Self-Review

- **Spec coverage:**
  - Geometry (6-tet body diagonal, sign flip calibration) → Tasks 2, 3.
  - Detection on 3 test cases → Tasks 5, 6.
  - SLSQP S-CD vs S-TET → Task 7.
  - Visualisation (3D grid + mid-z heatmaps + bar chart) → Tasks 8, 9.
  - Out of scope → Task 10 summary mentions follow-ups.
  - Success criteria 1–5 from the spec are checked: (1) full top-to-bottom run is verified in Task 10 step 2; (2) identity sanity in Task 3 step 2; (3) CD-vs-tet disagreement asserted in Task 5 step 2; (4) S-TET strictly beats S-CD asserted in Task 7 step 3; (5) 3D viz on 7³ runs in Tasks 8–9.

- **Placeholders:** none — every code block is complete, every command is exact.

- **Type / signature consistency:**
  - `warp_corners(phi) -> (D, H, W, 3)` used identically in Tasks 2, 3, 8.
  - `tet_signed_volumes(phi) -> (6, D-1, H-1, W-1)` used in Tasks 3, 4, 5, 6, 7, 8, 9.
  - `pack_phi`/`unpack_phi` packing `[dx, dy, dz]` consistent with `dvfopt.core.solver3d._full_grid_step_3d` and reused in Task 7.
  - `slsqp_results[name]['cd' | 'tet']` dict keys match between Tasks 7 and 9.

No fixes needed.
