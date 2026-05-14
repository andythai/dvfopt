"""Full-volume 3D benchmark runner.

Mirrors notebook 04_benchmark_3d_real_full.ipynb's outer L2 loop +
L1 polish phase as a stand-alone script. Each per-window SLSQP solve
runs in a child process with a wall-clock timeout, exactly as in 2D.

Status caveat: the windowed 3D-tet solver on this real DVF is
limited -- many folded cells are clustered in regions larger than a
single SLSQP-tractable window, and frozen-boundary cells leave the
local QP without enough degrees of freedom. The runner is included for
completeness and will produce a partial trajectory CSV; do not expect
full convergence on the whole volume out-of-the-box.
"""

import os
import sys
import time
import multiprocessing as mp

import numpy as np
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

MAX_WINDOW_HALF = 3              # window = (2H+1)^3 cells, centered on worst voxel
L2_MAX_OUTER = 5000
L2_MAX_MINIMIZE_ITER = 250
L2_LOCAL_TIMEOUT_S = 30
L2_STALL_LIMIT = 50

L1_POLISH_MAX_ITER = 200
L1_LOCAL_TIMEOUT_S = 20
L1_TOUCH_TOL = 1e-3
L1_COMPONENT_MAX_CELLS = 200

CHECKPOINT_EVERY = 50

OUTPUT_DIR = os.path.join(_HERE, 'output', '3d_real_full')
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAJ_PATH = os.path.join(OUTPUT_DIR, 'trajectory.csv')
CKPT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint.npz')
LOG_PATH = os.path.join(OUTPUT_DIR, 'run.log')

DATA_PATH = os.path.join(_REPO_ROOT, 'data',
                         'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')

TRAJ_COLUMNS = [
    'outer_iter', 'stage',
    'win_z0', 'win_z1', 'win_y0', 'win_y1', 'win_x0', 'win_x1',
    'win_cells', 'win_interior_cells',
    'slsqp_nit', 'slsqp_success', 'slsqp_status', 'slsqp_timed_out',
    'n_neg_tet_before', 'n_neg_tet_after',
    'min_tet_before', 'min_tet_after',
    'L1_before', 'L1_after', 'L2_before', 'L2_after',
    't_seconds',
]


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
    cz, cy, cx = focus_cell
    Dc, Hc, Wc = grid_shape
    z0 = max(0, cz - half_size); z1 = min(Dc, cz + half_size + 1)
    y0 = max(0, cy - half_size); y1 = min(Hc, cy + half_size + 1)
    x0 = max(0, cx - half_size); x1 = min(Wc, cx + half_size + 1)
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def init_csv():
    if not os.path.exists(TRAJ_PATH):
        with open(TRAJ_PATH, 'w') as f:
            f.write(','.join(TRAJ_COLUMNS) + '\n')


def append_row(row):
    parts = []
    for c in TRAJ_COLUMNS:
        v = row.get(c, '')
        if v is None:
            parts.append('')
        elif isinstance(v, float):
            parts.append(f'{v:.6g}')
        elif isinstance(v, bool):
            parts.append('True' if v else 'False')
        else:
            parts.append(str(v))
    with open(TRAJ_PATH, 'a') as f:
        f.write(','.join(parts) + '\n')


def save_checkpoint(arr, outer_iter):
    tmp = CKPT_PATH + '.tmp'
    np.savez_compressed(tmp, phi_corrected=arr, outer_iter=outer_iter)
    os.replace(tmp, CKPT_PATH)


def load_checkpoint(default):
    if not os.path.exists(CKPT_PATH):
        return default.copy(), 0
    try:
        with np.load(CKPT_PATH) as data:
            arr = data['phi_corrected']
            oi = int(data['outer_iter']) if 'outer_iter' in data.files else 0
            if arr.shape == default.shape:
                return arr.copy(), oi
    except Exception:
        pass
    return default.copy(), 0


def log_line(msg):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def main():
    log_line(f'[start] {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}')
    phi_full = np.load(DATA_PATH)
    log_line(f'shape: {phi_full.shape}')
    D, H, W = phi_full.shape[1:]
    init_csv()
    phi, start_outer = load_checkpoint(phi_full)
    phi_anchor = phi_full.copy()
    log_line(f'resume_from_outer={start_outer}')

    V0 = tet_signed_volumes(phi)
    log_line(f'init n_neg_tet={int((V0<=0).sum())}  min_tet={float(V0.min()):+.4f}'
              f'  cells_folded={int((V0.min(axis=0)<=0).sum())}')

    last_n_neg = None
    stall = 0
    outer_iter = start_outer
    t_run = time.time()
    while outer_iter < L2_MAX_OUTER:
        V = tet_signed_volumes(phi)
        n_neg = int((V <= 0).sum())
        if n_neg == 0:
            log_line(f'[L2-done] outer={outer_iter}  n_neg_tet=0')
            break
        cell_min = V.min(axis=0)
        argmin_flat = int(cell_min.argmin())
        cz, cy, cx = np.unravel_index(argmin_flat, cell_min.shape)
        z0, z1, y0, y1, x0, x1 = centered_window(
            (cz, cy, cx), half_size=MAX_WINDOW_HALF,
            grid_shape=cell_min.shape)
        sz, sy, sx = z1-z0, y1-y0, x1-x0
        win_cells = sz*sy*sx
        vsz, vsy, vsx = sz+1, sy+1, sx+1
        vz = slice(z0, z1+1); vy = slice(y0, y1+1); vx = slice(x0, x1+1)
        phi_win = phi[:, vz, vy, vx].copy()
        phi_anchor_win = phi_anchor[:, vz, vy, vx].copy()
        interior_mask = np.zeros((vsz, vsy, vsx), dtype=bool)
        if vsz > 2 and vsy > 2 and vsx > 2:
            interior_mask[1:-1, 1:-1, 1:-1] = True
        win_interior = int(interior_mask.sum())
        t0 = time.time()
        L1b = float(np.abs(phi - phi_anchor).sum())
        L2b = float(np.linalg.norm(phi - phi_anchor))
        min_tet_before = float(V.min())
        status, payload, info = _run_worker_with_timeout(
            _bench_worker.local_l2_3d_worker,
            (phi_win, phi_anchor_win, interior_mask,
             THRESHOLD, L2_MAX_MINIMIZE_ITER),
            timeout_s=L2_LOCAL_TIMEOUT_S)
        elapsed = time.time() - t0
        timed_out = status == 'timeout'
        if status == 'ok':
            phi_new, win_info = payload
            phi[:, vz, vy, vx] = phi_new
            slsqp_nit = win_info['nit']
            slsqp_success = win_info['success']
            slsqp_status = win_info['status']
        else:
            slsqp_nit, slsqp_success, slsqp_status = 0, False, -1
        V2 = tet_signed_volumes(phi)
        n_neg_after = int((V2 <= 0).sum())
        min_tet_after = float(V2.min())
        L1a = float(np.abs(phi - phi_anchor).sum())
        L2a = float(np.linalg.norm(phi - phi_anchor))
        row = {
            'outer_iter': outer_iter,
            'stage': 'L2_timeout' if timed_out else 'L2',
            'win_z0': z0, 'win_z1': z1, 'win_y0': y0, 'win_y1': y1,
            'win_x0': x0, 'win_x1': x1,
            'win_cells': win_cells, 'win_interior_cells': win_interior,
            'slsqp_nit': slsqp_nit, 'slsqp_success': slsqp_success,
            'slsqp_status': slsqp_status, 'slsqp_timed_out': timed_out,
            'n_neg_tet_before': n_neg, 'n_neg_tet_after': n_neg_after,
            'min_tet_before': min_tet_before, 'min_tet_after': min_tet_after,
            'L1_before': L1b, 'L1_after': L1a,
            'L2_before': L2b, 'L2_after': L2a,
            't_seconds': elapsed,
        }
        append_row(row)
        if last_n_neg is None or n_neg_after < last_n_neg:
            stall = 0
        else:
            stall += 1
        last_n_neg = n_neg_after
        if outer_iter % 25 == 0:
            log_line(f'[L2] outer={outer_iter:>5d}  n_neg {n_neg}->{n_neg_after}'
                      f'  min_tet {min_tet_before:+.3f}->{min_tet_after:+.3f}'
                      f'  win={sz}x{sy}x{sx}  slsqp_nit={slsqp_nit} ok={slsqp_success}'
                      f'  t={elapsed:.1f}s  stall={stall}  elapsed={time.time()-t_run:.0f}s')
        if outer_iter % CHECKPOINT_EVERY == 0 and outer_iter > start_outer:
            save_checkpoint(phi, outer_iter)
        if stall >= L2_STALL_LIMIT:
            log_line(f'[L2-stall] outer={outer_iter}  no progress for {stall} iters, exiting L2 loop')
            break
        outer_iter += 1

    save_checkpoint(phi, outer_iter)
    log_line(f'[L2-end] outer={outer_iter}  n_neg_tet={int((tet_signed_volumes(phi)<=0).sum())}'
              f'  elapsed={time.time()-t_run:.0f}s')

    # L1 polish phase (only meaningful if we got reasonably close to feasibility)
    V = tet_signed_volumes(phi)
    if int((V <= 0).sum()) > 0:
        log_line(f'[L1-skip] residual folds present; skipping L1 polish')
    else:
        log_line(f'[L1] starting L1 polish phase')
        diff_mag = np.abs(phi - phi_anchor).max(axis=0)
        Dc = diff_mag.shape[0] - 1
        Hc = diff_mag.shape[1] - 1
        Wc = diff_mag.shape[2] - 1
        cell_touched = np.zeros((Dc, Hc, Wc), dtype=bool)
        for (oz, oy, ox) in _CUBE_CORNERS:
            cell_touched |= diff_mag[oz:Dc+oz, oy:Hc+oy, ox:Wc+ox] > L1_TOUCH_TOL
        labels, n_comp = cc_label(cell_touched)
        log_line(f'[L1] {n_comp} touched components')
        accepted = 0
        for cid in range(1, n_comp + 1):
            comp = np.argwhere(labels == cid)
            wz0, wy0, wx0 = comp.min(axis=0)
            wz1, wy1, wx1 = comp.max(axis=0) + 1
            wz0 = max(0, wz0-1); wy0 = max(0, wy0-1); wx0 = max(0, wx0-1)
            wz1 = min(Dc, wz1+1); wy1 = min(Hc, wy1+1); wx1 = min(Wc, wx1+1)
            sz = wz1-wz0; sy = wy1-wy0; sx = wx1-wx0
            if sz*sy*sx > L1_COMPONENT_MAX_CELLS:
                continue
            vz = slice(wz0, wz1+1); vy = slice(wy0, wy1+1); vx = slice(wx0, wx1+1)
            phi_win = phi[:, vz, vy, vx].copy()
            phi_anchor_win = phi_anchor[:, vz, vy, vx].copy()
            vsz, vsy, vsx = sz+1, sy+1, sx+1
            interior = np.zeros((vsz, vsy, vsx), dtype=bool)
            if vsz > 2 and vsy > 2 and vsx > 2:
                interior[1:-1, 1:-1, 1:-1] = True
            else:
                continue
            status, payload, info = _run_worker_with_timeout(
                _bench_worker.local_l1_3d_worker,
                (phi_win, phi_anchor_win, interior, THRESHOLD, EPS_L1, L1_POLISH_MAX_ITER),
                timeout_s=L1_LOCAL_TIMEOUT_S)
            if status != 'ok':
                continue
            phi_new, win_info = payload
            candidate = phi.copy()
            candidate[:, vz, vy, vx] = phi_new
            Vc = tet_signed_volumes(candidate)
            if int((Vc <= 0).sum()) > 0:
                continue
            l1_before = float(np.abs(phi[:, vz, vy, vx] - phi_anchor_win).sum())
            l1_after = float(np.abs(phi_new - phi_anchor_win).sum())
            if l1_after < l1_before - 1e-9:
                phi = candidate
                accepted += 1
                if accepted % 20 == 0:
                    save_checkpoint(phi, outer_iter)
                    log_line(f'[L1] accepted {accepted}/{n_comp}')
        log_line(f'[L1-end] accepted {accepted}/{n_comp} components')

    save_checkpoint(phi, outer_iter)
    Vf = tet_signed_volumes(phi)
    log_line(f'[done] final n_neg_tet={int((Vf<=0).sum())}'
              f'  min_tet={float(Vf.min()):+.4f}'
              f'  L1={float(np.abs(phi - phi_anchor).sum()):.2f}'
              f'  L2={float(np.linalg.norm(phi - phi_anchor)):.2f}'
              f'  elapsed={time.time()-t_run:.0f}s')


if __name__ == '__main__':
    main()
