"""Smoke test: 3D windowed L2 outer loop with center-on-worst-cell strategy.

Replaces the "component bbox" window with a fixed-size cube centered
on the worst-fold cell. Keeps the worst cell well inside the interior
(away from the frozen boundary) so SLSQP actually has degrees of
freedom to fix it.
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
from _bench_worker import tet_signed_volumes

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


def centered_window(focus_cell, half_size, grid_shape):
    """Cell-bbox centered on `focus_cell` with `half_size` cells of border
    on each side. Clamps to the grid. Returns (z0,z1,y0,y1,x0,x1).
    """
    cz, cy, cx = focus_cell
    Dc, Hc, Wc = grid_shape
    z0 = max(0, cz - half_size); z1 = min(Dc, cz + half_size + 1)
    y0 = max(0, cy - half_size); y1 = min(Hc, cy + half_size + 1)
    x0 = max(0, cx - half_size); x1 = min(Wc, cx + half_size + 1)
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def smoke_run(half_size, max_outer, max_minimize_iter, local_timeout_s,
              roi):
    print(f'\n=== smoke run: half_size={half_size}  max_outer={max_outer} ===',
          flush=True)
    phi_full = np.load(DATA_PATH)
    z0,z1,y0,y1,x0,x1 = roi
    phi = phi_full[:, z0:z1, y0:y1, x0:x1].copy()
    phi_anchor = phi.copy()
    print(f'ROI shape: {phi.shape}', flush=True)
    V = tet_signed_volumes(phi)
    n_neg = int((V <= 0).sum())
    print(f'Initial: n_neg_tet={n_neg}  min_tet={float(V.min()):+.4f}'
          f'  cells_folded={int((V.min(axis=0)<=0).sum())}', flush=True)
    if n_neg == 0:
        return
    Dc, Hc, Wc = V.shape[1:]
    last_n_neg = None
    stall = 0
    t_total = 0.0
    for outer in range(max_outer):
        V = tet_signed_volumes(phi)
        n_neg = int((V <= 0).sum())
        if n_neg == 0:
            print(f'  outer={outer}  DONE n_neg=0', flush=True)
            break
        # Worst-fold cell.
        cell_min = V.min(axis=0)
        argmin_flat = int(cell_min.argmin())
        cz, cy, cx = np.unravel_index(argmin_flat, cell_min.shape)
        wz0, wz1, wy0, wy1, wx0, wx1 = centered_window(
            (cz, cy, cx), half_size=half_size, grid_shape=cell_min.shape)
        sz, sy, sx = wz1-wz0, wy1-wy0, wx1-wx0
        vz = slice(wz0, wz1+1); vy = slice(wy0, wy1+1); vx = slice(wx0, wx1+1)
        phi_win = phi[:, vz, vy, vx].copy()
        phi_anchor_win = phi_anchor[:, vz, vy, vx].copy()
        vsz, vsy, vsx = sz+1, sy+1, sx+1
        interior = np.zeros((vsz, vsy, vsx), dtype=bool)
        if vsz > 2 and vsy > 2 and vsx > 2:
            interior[1:-1, 1:-1, 1:-1] = True
        t0 = time.time()
        status, payload, info = _run_worker_with_timeout(
            _bench_worker.local_l2_3d_worker,
            (phi_win, phi_anchor_win, interior, THRESHOLD, max_minimize_iter),
            timeout_s=local_timeout_s)
        elapsed = time.time() - t0
        t_total += elapsed
        if status == 'ok':
            phi_new, info_w = payload
            phi[:, vz, vy, vx] = phi_new
            V2 = tet_signed_volumes(phi)
            n_neg_after = int((V2 <= 0).sum())
            min_tet_after = float(V2.min())
            if outer % 5 == 0 or n_neg_after != n_neg:
                print(f'  outer={outer:>3d}  win={sz}x{sy}x{sx} (centered on {cz},{cy},{cx})  '
                      f'n_neg {n_neg}->{n_neg_after}  '
                      f'min_tet {float(V.min()):+.3f}->{min_tet_after:+.3f}  '
                      f'slsqp_nit={info_w["nit"]} ok={info_w["success"]}  t={elapsed:.1f}s',
                      flush=True)
        elif status == 'timeout':
            print(f'  outer={outer:>3d}  TIMEOUT after {elapsed:.1f}s', flush=True)
        else:
            print(f'  outer={outer:>3d}  ERR {payload[0][:120]}', flush=True)
        T = tet_signed_volumes(phi)
        n_now = int((T <= 0).sum())
        if last_n_neg is None or n_now < last_n_neg:
            stall = 0
        else:
            stall += 1
        last_n_neg = n_now
        if stall >= 20:
            print(f'  STALLED at outer={outer}  n_neg={n_now}', flush=True)
            break
    V = tet_signed_volumes(phi)
    n_neg_final = int((V <= 0).sum())
    print(f'\nDONE  final n_neg={n_neg_final}  total t={t_total:.1f}s'
          f'  L1={float(np.abs(phi-phi_anchor).sum()):.2f}', flush=True)


def main():
    # Deeper ROI so the worst-fold cell isn't always pinned to a z-edge.
    # 30 slices is enough to host half_size=5 windows with room to spare;
    # the y/x extent of 80x100 keeps the cell count modest.
    roi = (110, 140, 100, 180, 100, 200)
    for half_size in [4, 6]:
        smoke_run(half_size=half_size,
                  max_outer=80, max_minimize_iter=300,
                  local_timeout_s=30, roi=roi)


if __name__ == '__main__':
    main()
