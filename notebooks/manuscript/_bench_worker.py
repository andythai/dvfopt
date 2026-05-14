"""Subprocess workers for the manuscript benchmark notebooks.

The 2D notebook calls ``iterative_serial`` inside a child process so a hung
SLSQP inner QP cannot block forward progress. The 3D notebook calls
``solve_local_*`` in the same way per outer-loop window.

Workers must live in an importable module (not a notebook cell) because
Windows ``multiprocessing`` uses spawn -- the child re-imports the target.
"""

import os
import sys

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

# Make sure dvfopt is importable from inside the child process. The
# notebooks insert ``../..`` on sys.path; do the same defensively here so
# the worker survives different working dirs.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def l2_iterative_worker(deformation, kwargs, send):
    """Child-process entry: run iterative_serial and send the result.

    Parameters
    ----------
    deformation : ndarray, shape (3, 1, H, W)
    kwargs : dict
        Forwarded to ``iterative_serial``.
    send : multiprocessing.Connection
        Pipe back to the parent. Sends ``('ok', phi)`` or ``('err', str)``.
    """
    try:
        from dvfopt import iterative_serial
        phi = iterative_serial(deformation, **kwargs)
        send.send(('ok', np.ascontiguousarray(phi)))
    except Exception as exc:  # noqa: BLE001
        import traceback
        send.send(('err', f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}'))
    finally:
        send.close()


# ============================================================
# 3D building blocks (replicated from the notebook so the child has them
# without depending on notebook-cell state).
# ============================================================
_CUBE_CORNERS = np.array([
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
], dtype=np.int8)
_TET_INDICES = np.array([
    [0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7],
    [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7],
], dtype=np.int8)


def _warp_corners(phi):
    D, H, W = phi.shape[1:]
    zz, yy, xx = np.mgrid[:D, :H, :W]
    return np.stack([xx + phi[2], yy + phi[1], zz + phi[0]], axis=-1)


def _tet_volumes_unsigned(corners):
    cell_corners = []
    for (oz, oy, ox) in _CUBE_CORNERS:
        oz, oy, ox = int(oz), int(oy), int(ox)  # avoid int8 overflow in slice arithmetic
        cell_corners.append(corners[oz:corners.shape[0] - 1 + oz,
                                    oy:corners.shape[1] - 1 + oy,
                                    ox:corners.shape[2] - 1 + ox])
    cell_corners = np.stack(cell_corners, axis=0)
    raw = np.empty((6,) + cell_corners.shape[1:-1], dtype=corners.dtype)
    for ti, (ia, ib, ic, id_) in enumerate(_TET_INDICES):
        a = cell_corners[ia]; b = cell_corners[ib]
        c = cell_corners[ic]; d = cell_corners[id_]
        ab = b - a; ac = c - a; ad = d - a
        cx = ac[..., 1] * ad[..., 2] - ac[..., 2] * ad[..., 1]
        cy = ac[..., 2] * ad[..., 0] - ac[..., 0] * ad[..., 2]
        cz = ac[..., 0] * ad[..., 1] - ac[..., 1] * ad[..., 0]
        raw[ti] = ab[..., 0] * cx + ab[..., 1] * cy + ab[..., 2] * cz
    return raw


_REF_SIGN = np.sign(_tet_volumes_unsigned(_warp_corners(np.zeros((3, 2, 2, 2))))[:, 0, 0, 0]).astype(np.float64)


def tet_signed_volumes(phi):
    raw = _tet_volumes_unsigned(_warp_corners(phi))
    return _REF_SIGN[:, None, None, None] * raw / 6.0


def _interior_pack_unpack(phi_win, interior_mask):
    int_idx = np.argwhere(interior_mask)
    n_int = len(int_idx)
    iz, iy, ix = int_idx[:, 0], int_idx[:, 1], int_idx[:, 2]

    def pack(phi):
        return np.concatenate([
            phi[2][iz, iy, ix], phi[1][iz, iy, ix], phi[0][iz, iy, ix]
        ])

    def unpack(z, base):
        out = base.copy()
        out[2][iz, iy, ix] = z[:n_int]
        out[1][iz, iy, ix] = z[n_int:2*n_int]
        out[0][iz, iy, ix] = z[2*n_int:]
        return out

    return pack, unpack, n_int


def full_grid_l2_3d_worker(phi_crop, phi_anchor_crop, threshold,
                            max_iter, send):
    """Full-grid L2 SLSQP on a 3D *crop*: every voxel-corner is a variable
    (no frozen edges), constraint is per-tet >= threshold across the
    whole crop. Mirrors notebook 17/18's converge_to_zero_folds inner
    SLSQP call. Use for cropped sub-volumes where boundary motion is OK.
    """
    try:
        D, H, W = phi_crop.shape[1:]
        voxels = D * H * W

        def pack(phi):
            return np.concatenate([phi[2].flatten(), phi[1].flatten(),
                                    phi[0].flatten()])

        def unpack(z_flat):
            dx = z_flat[:voxels].reshape(D, H, W)
            dy = z_flat[voxels:2*voxels].reshape(D, H, W)
            dz = z_flat[2*voxels:].reshape(D, H, W)
            return np.stack([dz, dy, dx])

        z_init = pack(phi_crop)
        z_anchor = pack(phi_anchor_crop)
        if np.allclose(z_init, z_anchor):
            rng = np.random.default_rng(42)
            z_init = z_init + rng.normal(scale=1e-3, size=z_init.shape)

        def obj(z):
            d = z - z_anchor
            return 0.5 * float(np.dot(d, d)), d

        def constr(z):
            return tet_signed_volumes(unpack(z)).flatten()

        nl = NonlinearConstraint(constr, lb=threshold, ub=np.inf)
        res = minimize(obj, z_init, jac=True, method='SLSQP',
                       constraints=[nl],
                       options={'maxiter': max_iter, 'disp': False})
        info = {'nit': int(res.nit), 'success': bool(res.success),
                'status': int(res.status)}
        send.send(('ok', unpack(res.x), info))
    except Exception as exc:  # noqa: BLE001
        import traceback
        send.send(('err', f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}'))
    finally:
        send.close()


def full_grid_l1_3d_worker(phi_crop, phi_anchor_crop, threshold, eps,
                            max_iter, send):
    """Full-grid L1 polish on a 3D crop. Smoothed-L1 objective anchored
    at phi_anchor_crop, per-tet >= threshold constraint, no frozen edges.
    """
    try:
        D, H, W = phi_crop.shape[1:]
        voxels = D * H * W

        def pack(phi):
            return np.concatenate([phi[2].flatten(), phi[1].flatten(),
                                    phi[0].flatten()])

        def unpack(z_flat):
            dx = z_flat[:voxels].reshape(D, H, W)
            dy = z_flat[voxels:2*voxels].reshape(D, H, W)
            dz = z_flat[2*voxels:].reshape(D, H, W)
            return np.stack([dz, dy, dx])

        z_init = pack(phi_crop)
        z_anchor = pack(phi_anchor_crop)
        if np.allclose(z_init, z_anchor):
            rng = np.random.default_rng(42)
            z_init = z_init + rng.normal(scale=1e-3, size=z_init.shape)

        def obj(z):
            d = z - z_anchor
            s = np.sqrt(d * d + eps * eps)
            return float(s.sum()), d / s

        def constr(z):
            return tet_signed_volumes(unpack(z)).flatten()

        nl = NonlinearConstraint(constr, lb=threshold, ub=np.inf)
        res = minimize(obj, z_init, jac=True, method='SLSQP',
                       constraints=[nl],
                       options={'maxiter': max_iter, 'disp': False})
        info = {'nit': int(res.nit), 'success': bool(res.success),
                'status': int(res.status)}
        send.send(('ok', unpack(res.x), info))
    except Exception as exc:  # noqa: BLE001
        import traceback
        send.send(('err', f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}'))
    finally:
        send.close()


def local_l2_3d_worker(phi_win, phi_anchor_win, interior_mask,
                       threshold, max_iter, send):
    """Run SLSQP with L2 objective + per-tet >= threshold constraint."""
    try:
        pack, unpack, n_int = _interior_pack_unpack(phi_win, interior_mask)
        if n_int == 0:
            send.send(('ok', phi_win.copy(), {'nit': 0, 'success': True, 'status': 0}))
            return
        z_init = pack(phi_win)
        z_anchor = pack(phi_anchor_win)
        # Avoid SLSQP's zero-gradient start when phi_win == phi_anchor_win.
        if np.allclose(z_init, z_anchor):
            rng = np.random.default_rng(42)
            z_init = z_init + rng.normal(scale=1e-3, size=z_init.shape)

        def obj(z):
            d = z - z_anchor
            return 0.5 * float(np.dot(d, d)), d

        def constr(z):
            return tet_signed_volumes(unpack(z, phi_win)).flatten()

        nl = NonlinearConstraint(constr, lb=threshold, ub=np.inf)
        res = minimize(obj, z_init, jac=True, method='SLSQP', constraints=[nl],
                       options={'maxiter': max_iter, 'disp': False})
        info = {'nit': int(res.nit), 'success': bool(res.success),
                'status': int(res.status)}
        send.send(('ok', unpack(res.x, phi_win), info))
    except Exception as exc:  # noqa: BLE001
        import traceback
        send.send(('err', f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}'))
    finally:
        send.close()


def local_l1_3d_worker(phi_win, phi_anchor_win, interior_mask,
                       threshold, eps, max_iter, send):
    """Run SLSQP with smoothed-L1 objective + per-tet >= threshold constraint."""
    try:
        pack, unpack, n_int = _interior_pack_unpack(phi_win, interior_mask)
        if n_int == 0:
            send.send(('ok', phi_win.copy(), {'nit': 0, 'success': True, 'status': 0}))
            return
        z_init = pack(phi_win)
        z_anchor = pack(phi_anchor_win)
        if np.allclose(z_init, z_anchor):
            rng = np.random.default_rng(42)
            z_init = z_init + rng.normal(scale=1e-3, size=z_init.shape)

        def obj(z):
            d = z - z_anchor
            s = np.sqrt(d * d + eps * eps)
            return float(s.sum()), d / s

        def constr(z):
            return tet_signed_volumes(unpack(z, phi_win)).flatten()

        nl = NonlinearConstraint(constr, lb=threshold, ub=np.inf)
        res = minimize(obj, z_init, jac=True, method='SLSQP', constraints=[nl],
                       options={'maxiter': max_iter, 'disp': False})
        info = {'nit': int(res.nit), 'success': bool(res.success),
                'status': int(res.status)}
        send.send(('ok', unpack(res.x, phi_win), info))
    except Exception as exc:  # noqa: BLE001
        import traceback
        send.send(('err', f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}'))
    finally:
        send.close()


# ============================================================
# 2D L1 polish helper (analogous to local_l1_3d but in 2D)
# ============================================================
def _interior_pack_unpack_2d(phi_win, interior_mask):
    int_idx = np.argwhere(interior_mask)
    n_int = len(int_idx)
    iy, ix = int_idx[:, 0], int_idx[:, 1]

    def pack(phi):
        return np.concatenate([phi[0][iy, ix], phi[1][iy, ix]])

    def unpack(z, base):
        out = base.copy()
        out[0][iy, ix] = z[:n_int]
        out[1][iy, ix] = z[n_int:]
        return out

    return pack, unpack, n_int


def _make_2tri_jac_2d(phi_win, interior_mask):
    """Build a closure ``jac(z) -> dense (n_constr, n_vars)`` that
    returns the analytical Jacobian of the 2-triangle constraint at z.

    Provides this Jacobian to SLSQP turns the finite-difference column
    sweep (N+1 constraint evals per QP iter) into a single tight numpy
    expression -- usually 10-100x speedup on the QP loop for crops of
    a few hundred variables.

    Math: T1[y,x] = -0.5 * det([BL-TR, BR-TR]) depends on the 3 corners
    TR=(y, x+1), BL=(y+1, x), BR=(y+1, x+1). T2[y,x] depends on TL, BL,
    TR. Since corner_y = ref_y + dy and corner_x = ref_x + dx, the
    partial wrt dy equals the partial wrt corner_y; same for dx.

      dT1/dTR.x = +0.5 (BR.y - BL.y)
      dT1/dTR.y = +0.5 (BL.x - BR.x)
      dT1/dBL.x = +0.5 (TR.y - BR.y)
      dT1/dBL.y = +0.5 (BR.x - TR.x)
      dT1/dBR.x = +0.5 (BL.y - TR.y)
      dT1/dBR.y = +0.5 (TR.x - BL.x)
    T2 has the same form with (A,B,C) = (TL, BL, TR).
    """
    _, H, W = phi_win.shape
    Hc, Wc = H - 1, W - 1
    n_cells = Hc * Wc
    n_constr = 2 * n_cells

    int_idx = np.argwhere(interior_mask)
    n_int = len(int_idx)
    iy = int_idx[:, 0].copy()
    ix = int_idx[:, 1].copy()
    int_pos = np.full((H, W), -1, dtype=np.int64)
    int_pos[iy, ix] = np.arange(n_int)
    n_vars = 2 * n_int

    # Per-cell corner -> variable-column indices (computed once).
    cy_idx = np.arange(Hc, dtype=np.int64)[:, None]
    cx_idx = np.arange(Wc, dtype=np.int64)[None, :]
    col_TL_dy = int_pos[cy_idx,     cx_idx]
    col_TR_dy = int_pos[cy_idx,     cx_idx + 1]
    col_BL_dy = int_pos[cy_idx + 1, cx_idx]
    col_BR_dy = int_pos[cy_idx + 1, cx_idx + 1]
    col_TL_dx = np.where(col_TL_dy >= 0, col_TL_dy + n_int, -1)
    col_TR_dx = np.where(col_TR_dy >= 0, col_TR_dy + n_int, -1)
    col_BL_dx = np.where(col_BL_dy >= 0, col_BL_dy + n_int, -1)
    col_BR_dx = np.where(col_BR_dy >= 0, col_BR_dy + n_int, -1)

    rows_T1 = (cy_idx * Wc + cx_idx).astype(np.int64) * np.ones((Hc, Wc), dtype=np.int64)
    rows_T2 = rows_T1 + n_cells

    # Pre-flatten valid (row, col) pairs for each partial.
    # Each partial-write covers all cells; some columns are -1 (corner
    # not in interior). Precompute valid masks + flat row/col arrays.
    partials = []
    for rows_arr, col_arr, label in [
        (rows_T1, col_TR_dy, 'T1_TR_dy'),
        (rows_T1, col_TR_dx, 'T1_TR_dx'),
        (rows_T1, col_BL_dy, 'T1_BL_dy'),
        (rows_T1, col_BL_dx, 'T1_BL_dx'),
        (rows_T1, col_BR_dy, 'T1_BR_dy'),
        (rows_T1, col_BR_dx, 'T1_BR_dx'),
        (rows_T2, col_TL_dy, 'T2_TL_dy'),
        (rows_T2, col_TL_dx, 'T2_TL_dx'),
        (rows_T2, col_TR_dy, 'T2_TR_dy'),
        (rows_T2, col_TR_dx, 'T2_TR_dx'),
        (rows_T2, col_BL_dy, 'T2_BL_dy'),
        (rows_T2, col_BL_dx, 'T2_BL_dx'),
    ]:
        col_flat = col_arr.ravel()
        valid = col_flat >= 0
        partials.append({
            'label': label,
            'rows': rows_arr.ravel()[valid],
            'cols': col_flat[valid],
            'valid': valid,  # bool mask (n_cells,)
        })

    iy_local = iy
    ix_local = ix
    ref_y = np.arange(H, dtype=np.float64)[:, None]
    ref_x = np.arange(W, dtype=np.float64)[None, :]
    phi_base = phi_win.copy()

    def jac(z):
        # Reconstruct full phi from frozen base + interior vars.
        phi_base[0][iy_local, ix_local] = z[:n_int]
        phi_base[1][iy_local, ix_local] = z[n_int:]
        def_x = ref_x + phi_base[1]
        def_y = ref_y + phi_base[0]
        TL_x = def_x[:-1, :-1]; TL_y = def_y[:-1, :-1]
        TR_x = def_x[:-1, 1:];  TR_y = def_y[:-1, 1:]
        BL_x = def_x[1:,  :-1]; BL_y = def_y[1:,  :-1]
        BR_x = def_x[1:,  1:];  BR_y = def_y[1:,  1:]

        # Per-cell partial values (each shape (Hc, Wc)).
        # T1 = -0.5 det([B-A, C-A]) with (A,B,C) = (TR, BL, BR).
        dT1_TR_x = 0.5 * (BR_y - BL_y)
        dT1_TR_y = 0.5 * (BL_x - BR_x)
        dT1_BL_x = 0.5 * (TR_y - BR_y)
        dT1_BL_y = 0.5 * (BR_x - TR_x)
        dT1_BR_x = 0.5 * (BL_y - TR_y)
        dT1_BR_y = 0.5 * (TR_x - BL_x)
        # T2 with (A,B,C) = (TL, BL, TR).
        dT2_TL_x = 0.5 * (TR_y - BL_y)
        dT2_TL_y = 0.5 * (BL_x - TR_x)
        dT2_BL_x = 0.5 * (TL_y - TR_y)
        dT2_BL_y = 0.5 * (TR_x - TL_x)
        dT2_TR_x = 0.5 * (BL_y - TL_y)
        dT2_TR_y = 0.5 * (TL_x - BL_x)

        vals = [
            dT1_TR_y, dT1_TR_x, dT1_BL_y, dT1_BL_x, dT1_BR_y, dT1_BR_x,
            dT2_TL_y, dT2_TL_x, dT2_TR_y, dT2_TR_x, dT2_BL_y, dT2_BL_x,
        ]
        J = np.zeros((n_constr, n_vars), dtype=np.float64)
        for p, v in zip(partials, vals):
            J[p['rows'], p['cols']] = v.ravel()[p['valid']]
        return J

    return jac


def solve_cluster_inline(c, phi_win, phi_anchor_win,
                          threshold, eps,
                          l2_max_passes, l2_max_iter, l1_max_iter):
    """All-in-one 2D cluster solver: notebook-18-style multi-pass L2
    SLSQP + L1 polish, fully inline (no subprocess). Designed for use
    inside a ``concurrent.futures.ProcessPoolExecutor`` worker.

    Returns ``(cluster_row_dict, phi_l1_or_None)``. ``phi_l1`` has
    the same shape as ``phi_win``; the caller should splice **only the
    interior_mask corners** back into the global slice to avoid
    overwriting other clusters' edits in parallel mode.
    """
    import time as _time
    from dvfopt.jacobian.triangle_sign import _triangle_areas_2d as _tri

    t0 = _time.time()
    interior_mask = c['interior_mask']
    row = {
        'z': c.get('z', -1),
        'cluster_id': c.get('cluster_id', -1),
        'y0': c['y0'], 'y1': c['y1'], 'x0': c['x0'], 'x1': c['x1'],
        'crop_cells_y': c['crop_cells_y'],
        'crop_cells_x': c['crop_cells_x'],
        'component_cells': c['component_cells'],
        'skipped_too_large': c['skipped_too_large'],
    }
    if c['skipped_too_large']:
        row.update({'cluster_t': _time.time() - t0, 'feasible': False})
        return row, None

    T1, T2 = _tri(phi_win[0], phi_win[1])
    init_n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
    init_min_tri = float(min(T1.min(), T2.min()))
    row.update({'init_n_neg_tri': init_n_neg, 'init_min_tri': init_min_tri})

    if init_n_neg == 0:
        row.update({
            'after_l2_n_neg_tri': 0, 'after_l2_min_tri': init_min_tri,
            'after_l1_n_neg_tri': 0, 'after_l1_min_tri': init_min_tri,
            'l2_passes_run': 0, 'l2_total_nit': 0, 'l2_total_t': 0.0,
            'l2_any_timeout': False,
            'l1_nit': 0, 'l1_t': 0.0,
            'l1_polished': False, 'l1_timed_out': False,
            'cluster_t': _time.time() - t0, 'feasible': True,
        })
        return row, None

    pack, unpack, n_int = _interior_pack_unpack_2d(phi_win, interior_mask)
    if n_int == 0:
        # No movable corners -> can't fix anything.
        row.update({
            'after_l2_n_neg_tri': init_n_neg, 'after_l2_min_tri': init_min_tri,
            'after_l1_n_neg_tri': init_n_neg, 'after_l1_min_tri': init_min_tri,
            'l2_passes_run': 0, 'l2_total_nit': 0, 'l2_total_t': 0.0,
            'l2_any_timeout': False,
            'l1_nit': 0, 'l1_t': 0.0,
            'l1_polished': False, 'l1_timed_out': False,
            'cluster_t': _time.time() - t0, 'feasible': False,
        })
        return row, None

    z_anchor = pack(phi_anchor_win)

    def obj_l2(z):
        d = z - z_anchor
        return 0.5 * float(np.dot(d, d)), d

    def constr(z):
        phi = unpack(z, phi_win)
        T1, T2 = _tri(phi[0], phi[1])
        return np.concatenate([T1.flatten(), T2.flatten()])

    jac_func = _make_2tri_jac_2d(phi_win, interior_mask)
    nl = NonlinearConstraint(constr, lb=threshold, ub=np.inf, jac=jac_func)

    # --- L2 multi-pass with perturb-on-stall ----------------------------
    # Each pass: SLSQP from the current iterate. If a pass leaves n_neg
    # unchanged (SLSQP succeeded but found no actual improvement, or
    # hit a fixed point), perturb the iterate before the next pass so
    # SLSQP has a different starting point. After STALL_PERTURB_LIMIT
    # consecutive non-improving passes, give up.
    STALL_PERTURB_LIMIT = 3
    phi_work = phi_win.copy()
    l2_total_nit = 0
    l2_total_t = 0.0
    l2_passes_run = 0
    last_n_neg = init_n_neg
    stall_count = 0
    perturb_seed = 0
    for pass_idx in range(l2_max_passes):
        T1, T2 = _tri(phi_work[0], phi_work[1])
        cur_n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
        if cur_n_neg == 0:
            break
        z_init = pack(phi_work)
        # Perturb on stall (or first pass when z == anchor); helps SLSQP
        # escape fixed points.
        if pass_idx == 0:
            z_init = _seed_perturb(z_init, z_anchor)
        elif stall_count > 0:
            rng = np.random.default_rng(101 + perturb_seed)
            sigma = 0.005 * stall_count   # escalating perturbation
            z_init = z_init + rng.normal(scale=sigma, size=z_init.shape)
            perturb_seed += 1
        t_pass = _time.time()
        res = minimize(obj_l2, z_init, jac=True, method='SLSQP',
                        constraints=[nl],
                        options={'maxiter': l2_max_iter, 'disp': False})
        l2_total_t += _time.time() - t_pass
        l2_passes_run += 1
        phi_new = unpack(res.x, phi_work)
        T1_new, T2_new = _tri(phi_new[0], phi_new[1])
        new_n_neg = int((T1_new <= 0).sum() + (T2_new <= 0).sum())
        # Accept only if STRICTLY better than current state. Otherwise
        # revert and bump stall counter (next pass will perturb harder).
        if new_n_neg < cur_n_neg:
            phi_work = phi_new
            l2_total_nit += int(res.nit)
            last_n_neg = new_n_neg
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= STALL_PERTURB_LIMIT:
                break

    T1, T2 = _tri(phi_work[0], phi_work[1])
    after_l2_n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
    after_l2_min = float(min(T1.min(), T2.min()))

    # --- L1 polish (only if L2 reached feasibility) -----------------
    l1_nit = 0
    l1_t = 0.0
    l1_polished = False
    after_l1_n_neg = after_l2_n_neg
    after_l1_min = after_l2_min
    phi_l1 = phi_work
    if after_l2_n_neg == 0:
        z_init = pack(phi_work)

        def obj_l1(z):
            d = z - z_anchor
            s = np.sqrt(d * d + eps * eps)
            return float(s.sum()), d / s

        t_pass = _time.time()
        res = minimize(obj_l1, z_init, jac=True, method='SLSQP',
                        constraints=[nl],
                        options={'maxiter': l1_max_iter, 'ftol': 1e-9,
                                 'disp': False})
        l1_t = _time.time() - t_pass
        l1_nit = int(res.nit)
        phi_candidate = unpack(res.x, phi_work)
        T1c, T2c = _tri(phi_candidate[0], phi_candidate[1])
        n_neg_c = int((T1c <= 0).sum() + (T2c <= 0).sum())
        L1_l2 = float(np.abs(phi_work - phi_anchor_win).sum())
        L1_c = float(np.abs(phi_candidate - phi_anchor_win).sum())
        if n_neg_c == 0 and L1_c < L1_l2 - 1e-9:
            phi_l1 = phi_candidate
            after_l1_n_neg = n_neg_c
            after_l1_min = float(min(T1c.min(), T2c.min()))
            l1_polished = True

    row.update({
        'after_l2_n_neg_tri': after_l2_n_neg, 'after_l2_min_tri': after_l2_min,
        'l2_passes_run': l2_passes_run, 'l2_total_nit': l2_total_nit,
        'l2_total_t': l2_total_t, 'l2_any_timeout': False,
        'after_l1_n_neg_tri': after_l1_n_neg, 'after_l1_min_tri': after_l1_min,
        'l1_nit': l1_nit, 'l1_t': l1_t,
        'l1_polished': l1_polished, 'l1_timed_out': False,
        'cluster_t': _time.time() - t0,
        'feasible': bool(after_l1_n_neg == 0),
    })
    return row, phi_l1


def _seed_perturb(z_init, z_anchor, sigma=1e-3, seed=42):
    """If z_init is exactly z_anchor, add a tiny Gaussian perturbation so
    SLSQP starts off the objective's zero-gradient point. Notebook-14
    style reactive warm-start, applied to *every* SLSQP call regardless
    of whether the previous run failed.
    """
    if not np.allclose(z_init, z_anchor):
        return z_init
    rng = np.random.default_rng(seed)
    return z_init + rng.normal(scale=sigma, size=z_init.shape)


def full_grid_l2_2d_worker(phi_crop, phi_anchor_crop, threshold,
                            max_iter, send):
    """Full-grid 2D L2 SLSQP on a crop. Every voxel-corner is a variable
    (no frozen edges). Constraint: every 2-triangle area >= threshold.
    Mirrors notebook 17/18's converge_to_zero_folds inner SLSQP call.
    """
    try:
        from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
        _, H, W = phi_crop.shape
        n_pix = H * W

        def pack(phi):
            return np.concatenate([phi[0].flatten(), phi[1].flatten()])

        def unpack(z_flat):
            dy = z_flat[:n_pix].reshape(H, W)
            dx = z_flat[n_pix:].reshape(H, W)
            return np.stack([dy, dx])

        z_init = pack(phi_crop)
        z_anchor = pack(phi_anchor_crop)
        if np.allclose(z_init, z_anchor):
            rng = np.random.default_rng(42)
            z_init = z_init + rng.normal(scale=1e-3, size=z_init.shape)

        def obj(z):
            d = z - z_anchor
            return 0.5 * float(np.dot(d, d)), d

        def constr(z):
            phi = unpack(z)
            T1, T2 = _triangle_areas_2d(phi[0], phi[1])
            return np.concatenate([T1.flatten(), T2.flatten()])

        nl = NonlinearConstraint(constr, lb=threshold, ub=np.inf)
        res = minimize(obj, z_init, jac=True, method='SLSQP',
                       constraints=[nl],
                       options={'maxiter': max_iter, 'disp': False})
        info = {'nit': int(res.nit), 'success': bool(res.success),
                'status': int(res.status)}
        send.send(('ok', unpack(res.x), info))
    except Exception as exc:  # noqa: BLE001
        import traceback
        send.send(('err', f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}'))
    finally:
        send.close()


def full_grid_l1_2d_worker(phi_crop, phi_anchor_crop, threshold, eps,
                            max_iter, send):
    """Full-grid 2D smoothed-L1 SLSQP polish on a crop."""
    try:
        from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
        _, H, W = phi_crop.shape
        n_pix = H * W

        def pack(phi):
            return np.concatenate([phi[0].flatten(), phi[1].flatten()])

        def unpack(z_flat):
            dy = z_flat[:n_pix].reshape(H, W)
            dx = z_flat[n_pix:].reshape(H, W)
            return np.stack([dy, dx])

        z_init = pack(phi_crop)
        z_anchor = pack(phi_anchor_crop)

        def obj(z):
            d = z - z_anchor
            s = np.sqrt(d * d + eps * eps)
            return float(s.sum()), d / s

        def constr(z):
            phi = unpack(z)
            T1, T2 = _triangle_areas_2d(phi[0], phi[1])
            return np.concatenate([T1.flatten(), T2.flatten()])

        nl = NonlinearConstraint(constr, lb=threshold, ub=np.inf)
        res = minimize(obj, z_init, jac=True, method='SLSQP',
                       constraints=[nl],
                       options={'maxiter': max_iter, 'ftol': 1e-9, 'disp': False})
        info = {'nit': int(res.nit), 'success': bool(res.success),
                'status': int(res.status)}
        send.send(('ok', unpack(res.x), info))
    except Exception as exc:  # noqa: BLE001
        import traceback
        send.send(('err', f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}'))
    finally:
        send.close()


def local_l2_2d_worker(phi_win, phi_anchor_win, interior_mask,
                       threshold, max_iter, send):
    """2D L2 SLSQP with 2-triangle constraint, frozen-edge interior mask.

    Note: the analytical-Jacobian path (_make_2tri_jac_2d) was tried but
    the dense Jacobian-build cost (~25 ms/call for typical crops) exceeds
    the finite-difference cost on our small clusters. Sticking with FD.
    """
    try:
        from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
        pack, unpack, n_int = _interior_pack_unpack_2d(phi_win, interior_mask)
        if n_int == 0:
            send.send(('ok', phi_win.copy(), {'nit': 0, 'success': True, 'status': 0}))
            return
        z_init = pack(phi_win)
        z_anchor = pack(phi_anchor_win)
        z_init = _seed_perturb(z_init, z_anchor)

        def obj(z):
            d = z - z_anchor
            return 0.5 * float(np.dot(d, d)), d

        def constr(z):
            phi = unpack(z, phi_win)
            T1, T2 = _triangle_areas_2d(phi[0], phi[1])
            return np.concatenate([T1.flatten(), T2.flatten()])

        nl = NonlinearConstraint(constr, lb=threshold, ub=np.inf)
        res = minimize(obj, z_init, jac=True, method='SLSQP', constraints=[nl],
                       options={'maxiter': max_iter, 'disp': False})
        info = {'nit': int(res.nit), 'success': bool(res.success),
                'status': int(res.status)}
        send.send(('ok', unpack(res.x, phi_win), info))
    except Exception as exc:  # noqa: BLE001
        import traceback
        send.send(('err', f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}'))
    finally:
        send.close()


def local_l1_2d_worker(phi_win, phi_anchor_win, interior_mask,
                       threshold, eps, max_iter, send):
    """2D smoothed-L1 SLSQP with 2-triangle constraint."""
    try:
        from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
        pack, unpack, n_int = _interior_pack_unpack_2d(phi_win, interior_mask)
        if n_int == 0:
            send.send(('ok', phi_win.copy(), {'nit': 0, 'success': True, 'status': 0}))
            return
        z_init = pack(phi_win)
        z_anchor = pack(phi_anchor_win)
        z_init = _seed_perturb(z_init, z_anchor)

        def obj(z):
            d = z - z_anchor
            s = np.sqrt(d * d + eps * eps)
            return float(s.sum()), d / s

        def constr(z):
            phi = unpack(z, phi_win)
            T1, T2 = _triangle_areas_2d(phi[0], phi[1])
            return np.concatenate([T1.flatten(), T2.flatten()])

        nl = NonlinearConstraint(constr, lb=threshold, ub=np.inf)
        res = minimize(obj, z_init, jac=True, method='SLSQP', constraints=[nl],
                       options={'maxiter': max_iter, 'disp': False})
        info = {'nit': int(res.nit), 'success': bool(res.success),
                'status': int(res.status)}
        send.send(('ok', unpack(res.x, phi_win), info))
    except Exception as exc:  # noqa: BLE001
        import traceback
        send.send(('err', f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}'))
    finally:
        send.close()
