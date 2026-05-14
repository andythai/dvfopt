"""Smoke test: run the 3D windowed L2 outer loop on a small ROI to verify
the subprocess-per-window timeout works and we make per-iter progress.
"""

import os
import sys
import time

import numpy as np
import multiprocessing as mp
from scipy.ndimage import label as cc_label

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from dvfopt import DEFAULT_PARAMS
import _bench_worker
from _bench_worker import tet_signed_volumes, _CUBE_CORNERS

THRESHOLD = DEFAULT_PARAMS['threshold']
EPS_L1 = 1e-4

DATA_PATH = os.path.join(
    _REPO_ROOT, 'data', 'corrected_correspondences_count_touching',
    'registered_output', 'deformation3d.npy')


def _run_worker_with_timeout(target, args, timeout_s):
    ctx = mp.get_context('spawn')
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(*args, child_conn))
    proc.start()
    child_conn.close()
    t0 = time.time()
    if parent_conn.poll(timeout_s):
        msg = parent_conn.recv()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate(); proc.join(timeout=5)
        return msg[0], msg[1:], {'wall_time': time.time() - t0}
    proc.terminate()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill(); proc.join()
    return 'timeout', None, {'wall_time': time.time() - t0}


def _shrink_to_cap(z0, z1, y0, y1, x0, x1, cap_per_axis, cap_total):
    def shrink_axis(a0, a1, cap):
        if a1 - a0 <= cap:
            return a0, a1
        ctr = (a0 + a1) // 2
        half = cap // 2
        return max(a0, ctr - half), min(a1, ctr - half + cap)
    z0, z1 = shrink_axis(z0, z1, cap_per_axis)
    y0, y1 = shrink_axis(y0, y1, cap_per_axis)
    x0, x1 = shrink_axis(x0, x1, cap_per_axis)
    while (z1-z0)*(y1-y0)*(x1-x0) > cap_total:
        spans = [(z1-z0, 'z'), (y1-y0, 'y'), (x1-x0, 'x')]
        spans.sort(reverse=True)
        ax = spans[0][1]
        if ax == 'z' and z1 - z0 > 2:
            z1 -= 1
        elif ax == 'y' and y1 - y0 > 2:
            y1 -= 1
        elif ax == 'x' and x1 - x0 > 2:
            x1 -= 1
        else:
            break
    return z0, z1, y0, y1, x0, x1


def main():
    print('Loading data...', flush=True)
    phi_full = np.load(DATA_PATH)
    print(f'shape: {phi_full.shape}', flush=True)

    # Pick a small ROI around a folded region. Use slice z=126 area
    # (lightest folded slice from 2D scan) +- a few z's.
    roi = (124, 130, 100, 180, 100, 200)  # (z0,z1,y0,y1,x0,x1)
    z0,z1,y0,y1,x0,x1 = roi
    phi = phi_full[:, z0:z1, y0:y1, x0:x1].copy()
    phi_anchor = phi.copy()
    print(f'ROI shape: {phi.shape}  ({roi})', flush=True)

    V0 = tet_signed_volumes(phi)
    n_neg = int((V0 <= 0).sum())
    print(f'Initial: n_neg_tet={n_neg}  min_tet={float(V0.min()):+.4f}'
          f'  cells_folded={int((V0.min(axis=0)<=0).sum())}', flush=True)

    if n_neg == 0:
        print('ROI has no folds -- pick a different ROI.')
        return

    L2_TIMEOUT_S = 30
    MAX_OUTER = 30
    MAX_WINDOW_CELLS = 800
    MAX_WINDOW_PER_AXIS = 12
    MAX_MINIMIZE_ITER = 200

    total_t = 0.0
    for outer in range(MAX_OUTER):
        V = tet_signed_volumes(phi)
        n_neg = int((V <= 0).sum())
        if n_neg == 0:
            print(f'  outer={outer}  n_neg=0 -- DONE', flush=True)
            break
        cell_fold_mask = V.min(axis=0) <= THRESHOLD - 1e-9
        labels, _ = cc_label(cell_fold_mask)
        argmin_flat = int(V.argmin())
        ti, cz, cy, cx = np.unravel_index(argmin_flat, V.shape)
        cid = labels[cz, cy, cx]
        comp = np.argwhere(labels == cid)
        wz0, wy0, wx0 = comp.min(axis=0); wz1, wy1, wx1 = comp.max(axis=0) + 1
        Dc, Hc, Wc = labels.shape
        wz0 = max(0, wz0 - 1); wy0 = max(0, wy0 - 1); wx0 = max(0, wx0 - 1)
        wz1 = min(Dc, wz1 + 1); wy1 = min(Hc, wy1 + 1); wx1 = min(Wc, wx1 + 1)
        wz0, wz1, wy0, wy1, wx0, wx1 = _shrink_to_cap(
            wz0, wz1, wy0, wy1, wx0, wx1,
            cap_per_axis=MAX_WINDOW_PER_AXIS, cap_total=MAX_WINDOW_CELLS)
        sz, sy, sx = wz1-wz0, wy1-wy0, wx1-wx0
        vz = slice(wz0, wz1+1); vy = slice(wy0, wy1+1); vx = slice(wx0, wx1+1)
        phi_win = phi[:, vz, vy, vx].copy()
        phi_anchor_win = phi_anchor[:, vz, vy, vx].copy()
        interior = np.zeros((sz+1, sy+1, sx+1), dtype=bool)
        if sz+1 > 2 and sy+1 > 2 and sx+1 > 2:
            interior[1:-1, 1:-1, 1:-1] = True
        t0 = time.time()
        status, payload, info = _run_worker_with_timeout(
            _bench_worker.local_l2_3d_worker,
            (phi_win, phi_anchor_win, interior, THRESHOLD, MAX_MINIMIZE_ITER),
            timeout_s=L2_TIMEOUT_S)
        elapsed = time.time() - t0
        total_t += elapsed
        if status == 'ok':
            phi_new, info_w = payload
            phi[:, vz, vy, vx] = phi_new
            V2 = tet_signed_volumes(phi)
            n_neg_after = int((V2 <= 0).sum())
            print(f'  outer={outer:>2d}  win={sz}x{sy}x{sx}  '
                  f'n_neg {n_neg}->{n_neg_after}  '
                  f'min_tet {float(V.min()):+.3f}->{float(V2.min()):+.3f}  '
                  f'slsqp_nit={info_w["nit"]} ok={info_w["success"]}  t={elapsed:.1f}s',
                  flush=True)
        elif status == 'timeout':
            print(f'  outer={outer:>2d}  win={sz}x{sy}x{sx}  TIMEOUT after {elapsed:.1f}s'
                  f'  n_neg stays {n_neg}', flush=True)
        else:
            print(f'  outer={outer:>2d}  ERR  {payload[0][:100]}', flush=True)
    V = tet_signed_volumes(phi)
    n_neg_final = int((V <= 0).sum())
    print(f'\nDONE  final n_neg={n_neg_final}  total t={total_t:.1f}s', flush=True)


if __name__ == '__main__':
    main()
