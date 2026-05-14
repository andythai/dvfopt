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

        def obj(z):
            d = z - z_anchor
            s = np.sqrt(d * d + eps * eps)
            return float(s.sum()), d / s

        def constr(z):
            return tet_signed_volumes(unpack(z, phi_win)).flatten()

        nl = NonlinearConstraint(constr, lb=threshold, ub=np.inf)
        res = minimize(obj, z_init, jac=True, method='SLSQP', constraints=[nl],
                       options={'maxiter': max_iter, 'ftol': 1e-9, 'disp': False})
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


def local_l2_2d_worker(phi_win, phi_anchor_win, interior_mask,
                       threshold, max_iter, send):
    """2D L2 SLSQP with 2-triangle constraint, frozen-edge interior mask."""
    try:
        from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
        pack, unpack, n_int = _interior_pack_unpack_2d(phi_win, interior_mask)
        if n_int == 0:
            send.send(('ok', phi_win.copy(), {'nit': 0, 'success': True, 'status': 0}))
            return
        z_init = pack(phi_win)
        z_anchor = pack(phi_anchor_win)

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
                       options={'maxiter': max_iter, 'ftol': 1e-9, 'disp': False})
        info = {'nit': int(res.nit), 'success': bool(res.success),
                'status': int(res.status)}
        send.send(('ok', unpack(res.x, phi_win), info))
    except Exception as exc:  # noqa: BLE001
        import traceback
        send.send(('err', f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}'))
    finally:
        send.close()
