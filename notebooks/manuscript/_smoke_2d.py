"""Smoke test: refactored 2D windowed L2 outer loop + perturb-on-stall.

Mirrors the logic in the 03_benchmark_2d_real_full notebook so we can
verify the pipeline before kicking off the full 528-slice run.
"""

import os
import sys
import time

import numpy as np
import multiprocessing as mp
from scipy.ndimage import label as cc_label
from scipy.ndimage import binary_dilation

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from dvfopt import DEFAULT_PARAMS
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
import _bench_worker

THRESHOLD = DEFAULT_PARAMS['threshold']
ERR_TOL = DEFAULT_PARAMS['err_tol']

MAX_WINDOW_PER_AXIS = 14
MAX_WINDOW_CELLS = 200
L2_MAX_OUTER = 1000          # smoke budget; full run uses 5000
L2_MAX_MINIMIZE_ITER = 250
L2_LOCAL_TIMEOUT_S = 30
L2_STALL_LIMIT = 30
L2_PERTURB_SIGMA = 0.02
L2_MAX_PERTURB_ROUNDS = 2


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


def _shrink_to_cap_2d(y0, y1, x0, x1, cap_per_axis, cap_total):
    def shrink_axis(a0, a1, cap):
        if a1 - a0 <= cap:
            return a0, a1
        ctr = (a0 + a1) // 2
        half = cap // 2
        return max(a0, ctr - half), min(a1, ctr - half + cap)
    y0, y1 = shrink_axis(y0, y1, cap_per_axis)
    x0, x1 = shrink_axis(x0, x1, cap_per_axis)
    while (y1 - y0) * (x1 - x0) > cap_total:
        if (y1 - y0) >= (x1 - x0) and (y1 - y0) > 2:
            y1 -= 1
        elif (x1 - x0) > 2:
            x1 -= 1
        else:
            break
    return y0, y1, x0, x1


def windowed_l2_pass(phi, phi_anchor, *, max_outer, stall_limit):
    last_n_neg = None
    stall = 0
    n_succ = n_to = n_err = 0
    for outer in range(max_outer):
        T1, T2 = _triangle_areas_2d(phi[0], phi[1])
        n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
        if n_neg == 0:
            return phi, outer + 1, n_succ, n_to, n_err, False
        cell_min = np.minimum(T1, T2)
        cell_fold_mask = cell_min <= THRESHOLD - ERR_TOL
        if not cell_fold_mask.any():
            return phi, outer + 1, n_succ, n_to, n_err, False
        labels, _ = cc_label(cell_fold_mask)
        argmin_flat = int(cell_min.argmin())
        wy, wx = np.unravel_index(argmin_flat, cell_min.shape)
        cid = labels[wy, wx]
        comp = np.argwhere(labels == cid)
        y0, x0 = comp.min(axis=0)
        y1, x1 = comp.max(axis=0) + 1
        Hc, Wc = labels.shape
        y0 = max(0, y0 - 1); x0 = max(0, x0 - 1)
        y1 = min(Hc, y1 + 1); x1 = min(Wc, x1 + 1)
        y0, y1, x0, x1 = _shrink_to_cap_2d(y0, y1, x0, x1,
                                            MAX_WINDOW_PER_AXIS, MAX_WINDOW_CELLS)
        sy, sx = y1 - y0, x1 - x0
        vy = slice(y0, y1 + 1); vx = slice(x0, x1 + 1)
        phi_win = phi[:, vy, vx].copy()
        phi_anchor_win = phi_anchor[:, vy, vx].copy()
        vsy, vsx = sy + 1, sx + 1
        interior_mask = np.zeros((vsy, vsx), dtype=bool)
        if vsy > 2 and vsx > 2:
            interior_mask[1:-1, 1:-1] = True
        status, payload, info = _run_worker_with_timeout(
            _bench_worker.local_l2_2d_worker,
            (phi_win, phi_anchor_win, interior_mask, THRESHOLD, L2_MAX_MINIMIZE_ITER),
            timeout_s=L2_LOCAL_TIMEOUT_S)
        if status == 'ok':
            phi_new = payload[0]
            phi[:, vy, vx] = phi_new
            n_succ += 1
        elif status == 'timeout':
            n_to += 1
        else:
            n_err += 1
        T1, T2 = _triangle_areas_2d(phi[0], phi[1])
        n_now = int((T1 <= 0).sum() + (T2 <= 0).sum())
        if last_n_neg is None or n_now < last_n_neg:
            stall = 0
        else:
            stall += 1
        last_n_neg = n_now
        if outer % 50 == 0:
            print(f'    outer={outer:>4d}  n_neg={n_now}  win={sy}x{sx}  '
                  f'status={status}  stall={stall}', flush=True)
        if stall >= stall_limit:
            return phi, outer + 1, n_succ, n_to, n_err, True
    return phi, max_outer, n_succ, n_to, n_err, False


def smoke_one_slice(phi_full, z):
    phi = np.stack([phi_full[1, z].copy(), phi_full[2, z].copy()])
    phi_anchor = phi.copy()
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    init_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
    print(f'  z={z}: initial n_neg_tri={init_neg}  min_tri={float(min(T1.min(),T2.min())):+.4f}',
          flush=True)
    if init_neg == 0:
        return
    H_, W_ = phi.shape[1:]
    t_slice = time.time()
    for r in range(L2_MAX_PERTURB_ROUNDS + 1):
        print(f'  --- L2 round {r} ---', flush=True)
        phi, n_outer, n_succ, n_to, n_err, stalled = windowed_l2_pass(
            phi, phi_anchor, max_outer=L2_MAX_OUTER, stall_limit=L2_STALL_LIMIT)
        T1, T2 = _triangle_areas_2d(phi[0], phi[1])
        n_neg_now = int((T1 <= 0).sum() + (T2 <= 0).sum())
        print(f'    round {r} done: outer={n_outer}  succ={n_succ} to={n_to} err={n_err}'
              f'  n_neg={n_neg_now}  stalled={stalled}', flush=True)
        if n_neg_now == 0:
            break
        if not stalled or r == L2_MAX_PERTURB_ROUNDS:
            break
        cell_min = np.minimum(T1, T2)
        cell_fold_mask = cell_min <= THRESHOLD - ERR_TOL
        pmask = binary_dilation(cell_fold_mask, iterations=2)
        pixel_mask = np.zeros((H_, W_), dtype=bool)
        pixel_mask[:-1, :-1] |= pmask
        pixel_mask[1:, :-1] |= pmask
        pixel_mask[:-1, 1:] |= pmask
        pixel_mask[1:, 1:] |= pmask
        pixel_mask[0, :] = False; pixel_mask[-1, :] = False
        pixel_mask[:, 0] = False; pixel_mask[:, -1] = False
        rng = np.random.default_rng(12345 + r)
        noise = rng.normal(scale=L2_PERTURB_SIGMA, size=(2, int(pixel_mask.sum())))
        phi[0][pixel_mask] += noise[0]
        phi[1][pixel_mask] += noise[1]
        print(f'    perturbed {int(pixel_mask.sum())} pixels with sigma={L2_PERTURB_SIGMA}',
              flush=True)
    L1 = float(np.abs(phi - phi_anchor).sum())
    L2 = float(np.linalg.norm(phi - phi_anchor))
    print(f'  slice done in {time.time()-t_slice:.1f}s: n_neg={n_neg_now}'
          f'  L1={L1:.3f}  L2={L2:.3f}', flush=True)


def main():
    print('Loading data...', flush=True)
    phi_full = np.load(os.path.join(
        _REPO_ROOT, 'data', 'corrected_correspondences_count_touching',
        'registered_output', 'deformation3d.npy'))
    print(f'shape: {phi_full.shape}', flush=True)
    D = phi_full.shape[1]
    counts = np.zeros(D, dtype=np.int64)
    for z in range(D):
        T1, T2 = _triangle_areas_2d(phi_full[1, z], phi_full[2, z])
        counts[z] = int((T1 <= 0).sum() + (T2 <= 0).sum())
    folded = np.where(counts > 0)[0]
    order = np.argsort(counts[folded])
    picks = [int(folded[order[0]]),
             int(folded[order[len(order) // 2]]),
             int(folded[order[-1]])]
    print(f'smoke slices: {picks}  fold counts: {[int(counts[z]) for z in picks]}',
          flush=True)
    for z in picks:
        print(f'\n=== smoke slice z={z} ===', flush=True)
        smoke_one_slice(phi_full, z)


if __name__ == '__main__':
    main()
