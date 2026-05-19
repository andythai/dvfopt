"""Head-to-head: FD vs analytical-fresh-alloc vs analytical-preallocated
Jacobian on the same realistic cluster (one large cluster from z=200 of
the real data). All three run through the same subprocess + SLSQP path
so the comparison is apples-to-apples.
"""

import os
import sys
import time
import multiprocessing as mp

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from dvfopt import DEFAULT_PARAMS
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
import _bench_worker

THRESHOLD = DEFAULT_PARAMS['threshold']


def worker_fd(phi_win, phi_anchor, interior_mask, max_iter, send):
    """SLSQP with finite-difference Jacobian (scipy default)."""
    try:
        pack, unpack, n_int = _bench_worker._interior_pack_unpack_2d(phi_win, interior_mask)
        z_init = pack(phi_win)
        z_anchor = pack(phi_anchor)
        z_init = _bench_worker._seed_perturb(z_init, z_anchor)
        def obj(z):
            d = z - z_anchor
            return 0.5 * float(np.dot(d, d)), d
        def constr(z):
            phi = unpack(z, phi_win)
            T1, T2 = _triangle_areas_2d(phi[0], phi[1])
            return np.concatenate([T1.flatten(), T2.flatten()])
        nl = NonlinearConstraint(constr, lb=THRESHOLD, ub=np.inf)
        t0 = time.time()
        res = minimize(obj, z_init, jac=True, method='SLSQP', constraints=[nl],
                       options={'maxiter': max_iter, 'disp': False})
        send.send(('ok', unpack(res.x, phi_win), {
            't': time.time() - t0, 'nit': res.nit,
            'success': bool(res.success), 'status': int(res.status),
        }))
    except Exception as exc:
        send.send(('err', str(exc)))
    finally:
        send.close()


def worker_an_fresh(phi_win, phi_anchor, interior_mask, max_iter, send):
    """Analytical Jacobian, fresh np.zeros((n_constr, n_vars)) per call.
    (Matches the current _bench_worker behaviour.)"""
    try:
        pack, unpack, n_int = _bench_worker._interior_pack_unpack_2d(phi_win, interior_mask)
        z_init = pack(phi_win)
        z_anchor = pack(phi_anchor)
        z_init = _bench_worker._seed_perturb(z_init, z_anchor)
        def obj(z):
            d = z - z_anchor
            return 0.5 * float(np.dot(d, d)), d
        def constr(z):
            phi = unpack(z, phi_win)
            T1, T2 = _triangle_areas_2d(phi[0], phi[1])
            return np.concatenate([T1.flatten(), T2.flatten()])
        jac_func = _bench_worker._make_2tri_jac_2d(phi_win, interior_mask)
        nl = NonlinearConstraint(constr, lb=THRESHOLD, ub=np.inf, jac=jac_func)
        t0 = time.time()
        res = minimize(obj, z_init, jac=True, method='SLSQP', constraints=[nl],
                       options={'maxiter': max_iter, 'disp': False})
        send.send(('ok', unpack(res.x, phi_win), {
            't': time.time() - t0, 'nit': res.nit,
            'success': bool(res.success), 'status': int(res.status),
        }))
    except Exception as exc:
        send.send(('err', str(exc)))
    finally:
        send.close()


def _make_2tri_jac_2d_prealloc(phi_win, interior_mask):
    """Same analytical Jacobian as _make_2tri_jac_2d but with a single
    pre-allocated dense output buffer that gets J.fill(0.0)'d each call.
    Avoids the per-call np.zeros((n_constr, n_vars)) malloc.
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

    partials = []
    for rows_arr, col_arr in [
        (rows_T1, col_TR_dy), (rows_T1, col_TR_dx),
        (rows_T1, col_BL_dy), (rows_T1, col_BL_dx),
        (rows_T1, col_BR_dy), (rows_T1, col_BR_dx),
        (rows_T2, col_TL_dy), (rows_T2, col_TL_dx),
        (rows_T2, col_TR_dy), (rows_T2, col_TR_dx),
        (rows_T2, col_BL_dy), (rows_T2, col_BL_dx),
    ]:
        col_flat = col_arr.ravel()
        valid = col_flat >= 0
        partials.append({
            'rows': rows_arr.ravel()[valid],
            'cols': col_flat[valid],
            'valid': valid,
        })

    iy_local = iy
    ix_local = ix
    ref_y = np.arange(H, dtype=np.float64)[:, None]
    ref_x = np.arange(W, dtype=np.float64)[None, :]
    phi_base = phi_win.copy()
    # Pre-allocated dense Jacobian buffer (reused across calls).
    J = np.zeros((n_constr, n_vars), dtype=np.float64)

    def jac(z):
        phi_base[0][iy_local, ix_local] = z[:n_int]
        phi_base[1][iy_local, ix_local] = z[n_int:]
        def_x = ref_x + phi_base[1]
        def_y = ref_y + phi_base[0]
        TL_x = def_x[:-1, :-1]; TL_y = def_y[:-1, :-1]
        TR_x = def_x[:-1, 1:];  TR_y = def_y[:-1, 1:]
        BL_x = def_x[1:,  :-1]; BL_y = def_y[1:,  :-1]
        BR_x = def_x[1:,  1:];  BR_y = def_y[1:,  1:]
        dT1_TR_x = 0.5 * (BR_y - BL_y)
        dT1_TR_y = 0.5 * (BL_x - BR_x)
        dT1_BL_x = 0.5 * (TR_y - BR_y)
        dT1_BL_y = 0.5 * (BR_x - TR_x)
        dT1_BR_x = 0.5 * (BL_y - TR_y)
        dT1_BR_y = 0.5 * (TR_x - BL_x)
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
        J.fill(0.0)
        for p, v in zip(partials, vals):
            J[p['rows'], p['cols']] = v.ravel()[p['valid']]
        return J

    return jac


def worker_an_prealloc(phi_win, phi_anchor, interior_mask, max_iter, send):
    try:
        pack, unpack, n_int = _bench_worker._interior_pack_unpack_2d(phi_win, interior_mask)
        z_init = pack(phi_win)
        z_anchor = pack(phi_anchor)
        z_init = _bench_worker._seed_perturb(z_init, z_anchor)
        def obj(z):
            d = z - z_anchor
            return 0.5 * float(np.dot(d, d)), d
        def constr(z):
            phi = unpack(z, phi_win)
            T1, T2 = _triangle_areas_2d(phi[0], phi[1])
            return np.concatenate([T1.flatten(), T2.flatten()])
        jac_func = _make_2tri_jac_2d_prealloc(phi_win, interior_mask)
        nl = NonlinearConstraint(constr, lb=THRESHOLD, ub=np.inf, jac=jac_func)
        t0 = time.time()
        res = minimize(obj, z_init, jac=True, method='SLSQP', constraints=[nl],
                       options={'maxiter': max_iter, 'disp': False})
        send.send(('ok', unpack(res.x, phi_win), {
            't': time.time() - t0, 'nit': res.nit,
            'success': bool(res.success), 'status': int(res.status),
        }))
    except Exception as exc:
        send.send(('err', str(exc)))
    finally:
        send.close()


def _run_in_subproc(target, args, timeout_s):
    ctx = mp.get_context('spawn')
    pc, cc = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(*args, cc))
    proc.start(); cc.close()
    t0 = time.time()
    if pc.poll(timeout_s):
        msg = pc.recv()
        proc.join(timeout=5)
        if proc.is_alive(): proc.terminate(); proc.join()
        return msg, time.time() - t0
    proc.terminate(); proc.join()
    return ('timeout',), time.time() - t0


def build_cluster_crops_from_slice(phi_slice, phi_anchor_slice, n_clusters=20):
    """Build a representative set of cluster crops from the actual slice
    -- same interior_mask + bbox shape as the real runner.
    """
    from scipy.ndimage import label as cc_label, find_objects, binary_dilation
    T1, T2 = _triangle_areas_2d(phi_slice[0], phi_slice[1])
    cell_min = np.minimum(T1, T2)
    cell_fold_mask = cell_min <= 0
    dilated = binary_dilation(cell_fold_mask, iterations=1)
    labels, n_comp = cc_label(dilated)
    bboxes = find_objects(labels)
    Hc, Wc = cell_fold_mask.shape
    sizes = np.bincount(labels.ravel(), minlength=n_comp + 1)[1:]
    order = np.argsort(-sizes)

    crops = []
    for idx in order[:n_clusters]:
        bbox = bboxes[idx]
        if bbox is None: continue
        cy0, cy1 = bbox[0].start, bbox[0].stop
        cx0, cx1 = bbox[1].start, bbox[1].stop
        y0 = max(0, cy0 - 1); y1 = min(Hc, cy1 + 1)
        x0 = max(0, cx0 - 1); x1 = min(Wc, cx1 + 1)
        cluster_in_bbox = (labels[y0:y1, x0:x1] == idx + 1)
        sy = y1 - y0; sx = x1 - x0
        interior_mask = np.zeros((sy + 1, sx + 1), dtype=bool)
        for di in (0, 1):
            for dj in (0, 1):
                interior_mask[di:di + sy, dj:dj + sx] |= cluster_in_bbox
        vy = slice(y0, y1 + 1); vx = slice(x0, x1 + 1)
        crops.append({
            'phi_win': phi_slice[:, vy, vx].copy(),
            'phi_anchor': phi_anchor_slice[:, vy, vx].copy(),
            'interior_mask': interior_mask,
            'sy': sy, 'sx': sx, 'comp_cells': int(sizes[idx]),
        })
    return crops


def main():
    phi_full = np.load(os.path.join(_REPO_ROOT, 'data',
        'corrected_correspondences_count_touching',
        'registered_output', 'deformation3d.npy'))
    # Use z=200 -- it has 622 init folds, gives a mix of cluster sizes.
    z = 200
    phi_slice = np.stack([phi_full[1, z], phi_full[2, z]])
    phi_anchor_slice = phi_slice.copy()
    crops = build_cluster_crops_from_slice(phi_slice, phi_anchor_slice, n_clusters=15)
    print(f'sampled {len(crops)} clusters from z={z}')
    print(f'cluster sizes (cells, sy*sx): '
          f'{[(c["comp_cells"], c["sy"]*c["sx"]) for c in crops]}')

    variants = [
        ('FD',          worker_fd),
        ('AN_fresh',    worker_an_fresh),
        ('AN_prealloc', worker_an_prealloc),
    ]
    results = {name: [] for name, _ in variants}

    for ci, c in enumerate(crops):
        print(f'\n--- cluster {ci}: {c["sy"]}x{c["sx"]} bbox, {c["comp_cells"]} fold cells ---')
        for name, target in variants:
            msg, wall = _run_in_subproc(
                target, (c['phi_win'], c['phi_anchor'], c['interior_mask'], 80),
                timeout_s=120,
            )
            if msg[0] == 'ok':
                info = msg[2]
                results[name].append(wall)
                print(f'  {name:>13s}: wall={wall:6.2f}s  slsqp_t={info["t"]:6.2f}s  '
                      f'nit={info["nit"]:>3d}  ok={info["success"]}')
            else:
                results[name].append(None)
                print(f'  {name:>13s}: {msg}')

    print('\n=== SUMMARY (sum of wall times across all clusters) ===')
    for name, _ in variants:
        ts = [t for t in results[name] if t is not None]
        if ts:
            print(f'  {name:>13s}: total={sum(ts):7.2f}s  mean={np.mean(ts):.2f}s  '
                  f'n_succ={len(ts)}/{len(crops)}')
        else:
            print(f'  {name:>13s}: no successes')


if __name__ == '__main__':
    main()
