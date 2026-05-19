"""Full-volume 2D benchmark runner.

Mirrors notebook 03_benchmark_2d_real_full.ipynb's pipeline as a
stand-alone script so it can run unattended for many hours. Writes
per_slice.csv incrementally and snapshots checkpoint.npz every
CHECKPOINT_EVERY slices, so a kernel kill loses at most CHECKPOINT_EVERY
slices' worth of corrected-volume state (the CSV row for each finished
slice survives even without the snapshot).
"""

import os
import sys
import time
import traceback
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label
from scipy.ndimage import binary_dilation

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from dvfopt import DEFAULT_PARAMS
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
import _bench_worker

# ---------------------------------------------------------------------
# Tunables (kept in sync with notebook 03).
# ---------------------------------------------------------------------
THRESHOLD = DEFAULT_PARAMS['threshold']
ERR_TOL = DEFAULT_PARAMS['err_tol']
EPS_L1 = 1e-4

MAX_WINDOW_PER_AXIS = 14
MAX_WINDOW_CELLS = 200
L2_MAX_OUTER = 20000             # outer iters per perturb round
L2_MAX_MINIMIZE_ITER = 250
L2_LOCAL_TIMEOUT_S = 30
L2_STALL_LIMIT = 150             # consecutive non-improving iters before perturb
L2_PERTURB_SIGMA = 0.005         # smaller noise -- less damage L2 has to clean up
L2_MAX_PERTURB_ROUNDS = 8

L1_POLISH_MAX_ITER = 250
L1_LOCAL_TIMEOUT_S = 20
L1_TOUCH_TOL = 1e-3
L1_BORDER = 1
L1_COMPONENT_MAX_CELLS = 200

CHECKPOINT_EVERY = 10

OUTPUT_DIR = os.path.join(_HERE, 'output', '2d_real_full')
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, 'per_slice.csv')
CKPT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint.npz')
LOG_PATH = os.path.join(OUTPUT_DIR, 'run.log')

DATA_PATH = os.path.join(_REPO_ROOT, 'data',
                         'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')


CSV_COLUMNS = [
    'z', 'H', 'W',
    'init_n_neg_tri', 'init_min_tri',
    'L2_n_neg_tri', 'L2_min_tri', 'L2_L1', 'L2_L2', 'L2_t',
    'L2_outer_iters', 'L2_local_timeouts', 'L2_local_errors',
    'L2_perturb_rounds', 'L2_final_stalled', 'L2_exception',
    'L1_L1', 'L1_L2', 'L1_n_neg_tri', 'L1_min_tri',
    'L1_components', 'L1_polished', 'L1_rejected',
    'L1_skipped_large', 'L1_timed_out',
    'L1_iter_total', 'L1_t',
    'total_t', 'feasible', 'l1_drop_pct',
]


# ---------------------------------------------------------------------
# Subprocess timeout helper
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Window builder
# ---------------------------------------------------------------------
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


def _build_window_around_2d(focus_cell, labels, cell_fold_mask,
                             *, border=1,
                             cap_per_axis=MAX_WINDOW_PER_AXIS,
                             cap_total=MAX_WINDOW_CELLS):
    cy, cx = focus_cell
    cid = labels[cy, cx]
    if cid == 0:
        y0, y1 = cy, cy + 1
        x0, x1 = cx, cx + 1
    else:
        comp = np.argwhere(labels == cid)
        y0, x0 = comp.min(axis=0)
        y1, x1 = comp.max(axis=0) + 1
    Hc, Wc = labels.shape
    y0 = max(0, y0 - border); y1 = min(Hc, y1 + border)
    x0 = max(0, x0 - border); x1 = min(Wc, x1 + border)
    y0, y1, x0, x1 = _shrink_to_cap_2d(y0, y1, x0, x1, cap_per_axis, cap_total)
    return int(y0), int(y1), int(x0), int(x1)


# ---------------------------------------------------------------------
# Per-slice pipeline (mirrors notebook 03 cells)
# ---------------------------------------------------------------------
def _windowed_l2_pass(phi, phi_anchor, *, max_outer, max_minimize_iter,
                       local_timeout_s, stall_limit):
    """Strict-monotonic windowed outer loop.

    Each window solve is accepted only if it *strictly reduces* the
    global n_neg_tri count; otherwise the slice's phi is reverted. The
    function always returns the best phi (lowest n_neg) seen during the
    pass, so a stalled or noisy late iteration can never undo earlier
    progress.
    """
    stats = {'outer_iters_run': 0, 'local_timeouts': 0,
             'local_errors': 0, 'local_successes': 0,
             'local_no_progress': 0,
             'stalled': False, 'window_size_max': 0}
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    best_n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
    best_phi = phi.copy()
    stall = 0
    for outer in range(max_outer):
        T1, T2 = _triangle_areas_2d(phi[0], phi[1])
        cell_min = np.minimum(T1, T2)
        cell_fold_mask = cell_min <= THRESHOLD - ERR_TOL
        n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
        if n_neg == 0 or not cell_fold_mask.any():
            stats['n_neg_after'] = n_neg
            best_n_neg = min(best_n_neg, n_neg)
            best_phi = phi.copy() if n_neg <= best_n_neg else best_phi
            stats['n_neg_after'] = best_n_neg
            return best_phi, stats
        labels, _ = cc_label(cell_fold_mask)
        argmin_flat = int(cell_min.argmin())
        wy, wx = np.unravel_index(argmin_flat, cell_min.shape)
        y0, y1, x0, x1 = _build_window_around_2d(
            (wy, wx), labels, cell_fold_mask,
            cap_per_axis=MAX_WINDOW_PER_AXIS, cap_total=MAX_WINDOW_CELLS)
        sy, sx = y1 - y0, x1 - x0
        stats['window_size_max'] = max(stats['window_size_max'], sy * sx)
        vsy, vsx = sy + 1, sx + 1
        vy = slice(y0, y1 + 1); vx = slice(x0, x1 + 1)
        phi_win = phi[:, vy, vx].copy()
        phi_anchor_win = phi_anchor[:, vy, vx].copy()
        interior_mask = np.zeros((vsy, vsx), dtype=bool)
        if vsy > 2 and vsx > 2:
            interior_mask[1:-1, 1:-1] = True
        # Snapshot the slice (cheap -- only the window) so we can revert
        # if the solve didn't help globally.
        phi_win_pre = phi[:, vy, vx].copy()
        status, payload, info = _run_worker_with_timeout(
            _bench_worker.local_l2_2d_worker,
            (phi_win, phi_anchor_win, interior_mask, THRESHOLD, max_minimize_iter),
            timeout_s=local_timeout_s)
        if status == 'ok':
            phi[:, vy, vx] = payload[0]
            T1, T2 = _triangle_areas_2d(phi[0], phi[1])
            n_neg_now = int((T1 <= 0).sum() + (T2 <= 0).sum())
            if n_neg_now < best_n_neg:
                # Strictly better than the best so far -- accept and record.
                best_n_neg = n_neg_now
                best_phi = phi.copy()
                stats['local_successes'] += 1
                stall = 0
            else:
                # Solve succeeded but didn't strictly improve globally --
                # revert so noise can't accumulate.
                phi[:, vy, vx] = phi_win_pre
                stats['local_no_progress'] += 1
                stall += 1
        elif status == 'timeout':
            stats['local_timeouts'] += 1
            stall += 1
        else:
            stats['local_errors'] += 1
            stall += 1
        stats['outer_iters_run'] = outer + 1
        if best_n_neg == 0:
            stats['n_neg_after'] = 0
            return best_phi, stats
        if stall >= stall_limit:
            stats['stalled'] = True
            stats['n_neg_after'] = best_n_neg
            return best_phi, stats
    stats['n_neg_after'] = best_n_neg
    return best_phi, stats


def run_l2_phase(deformation_3channel):
    t_start = time.time()
    phi = np.stack([deformation_3channel[1, 0].copy(),
                     deformation_3channel[2, 0].copy()])
    phi_anchor = phi.copy()
    H_, W_ = phi.shape[1:]
    T1, T2 = _triangle_areas_2d(phi_anchor[0], phi_anchor[1])
    best_n_neg_global = int((T1 <= 0).sum() + (T2 <= 0).sum())
    best_phi_global = phi.copy()
    info = {'wall_time': 0.0, 'exception': None, 'timed_out': False,
            'outer_iters_total': 0, 'local_timeouts_total': 0,
            'local_errors_total': 0, 'local_successes_total': 0,
            'perturb_rounds': 0, 'final_stalled': False}
    try:
        for r in range(L2_MAX_PERTURB_ROUNDS + 1):
            pass_phi, stats = _windowed_l2_pass(
                phi, phi_anchor,
                max_outer=L2_MAX_OUTER,
                max_minimize_iter=L2_MAX_MINIMIZE_ITER,
                local_timeout_s=L2_LOCAL_TIMEOUT_S,
                stall_limit=L2_STALL_LIMIT)
            info['outer_iters_total'] += stats['outer_iters_run']
            info['local_timeouts_total'] += stats['local_timeouts']
            info['local_errors_total'] += stats['local_errors']
            info['local_successes_total'] += stats['local_successes']
            # Update best across rounds.
            if stats['n_neg_after'] < best_n_neg_global:
                best_n_neg_global = stats['n_neg_after']
                best_phi_global = pass_phi.copy()
            if best_n_neg_global == 0:
                break
            if not stats['stalled'] or r == L2_MAX_PERTURB_ROUNDS:
                break
            # Perturb FROM the best-seen state, so a bad round can't
            # cause the next round to start from a worse position.
            phi = best_phi_global.copy()
            T1, T2 = _triangle_areas_2d(phi[0], phi[1])
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
            n_pert = int(pixel_mask.sum())
            if n_pert > 0:
                noise = rng.normal(scale=L2_PERTURB_SIGMA, size=(2, n_pert))
                phi[0][pixel_mask] += noise[0]
                phi[1][pixel_mask] += noise[1]
                info['perturb_rounds'] += 1
    except Exception as exc:
        info['exception'] = f'{type(exc).__name__}: {exc}'
    info['final_stalled'] = bool(best_n_neg_global > 0)
    info['wall_time'] = time.time() - t_start
    return best_phi_global, info


def run_l1_polish_phase(phi_l2, phi_anchor):
    t0 = time.time()
    H_, W_ = phi_l2.shape[1:]
    diff_mag = np.abs(phi_l2 - phi_anchor).max(axis=0)
    touched = diff_mag > L1_TOUCH_TOL
    info = {'n_components': 0, 'n_components_polished': 0,
            'n_components_rejected': 0, 'n_components_skipped_too_large': 0,
            'n_components_timed_out': 0,
            'iter_total': 0, 'wall_time': 0.0}
    if not touched.any():
        info['wall_time'] = time.time() - t0
        return phi_l2.copy(), info
    labels, n_comp = cc_label(touched)
    info['n_components'] = int(n_comp)
    phi_l1 = phi_l2.copy()
    for cid in range(1, n_comp + 1):
        cells = np.argwhere(labels == cid)
        y0, x0 = cells.min(axis=0)
        y1, x1 = cells.max(axis=0) + 1
        y0 = max(0, y0 - L1_BORDER); x0 = max(0, x0 - L1_BORDER)
        y1 = min(H_, y1 + L1_BORDER); x1 = min(W_, x1 + L1_BORDER)
        win_h = y1 - y0; win_w = x1 - x0
        if win_h * win_w > L1_COMPONENT_MAX_CELLS or win_h < 3 or win_w < 3:
            info['n_components_skipped_too_large'] += 1
            continue
        interior_mask = np.zeros((win_h, win_w), dtype=bool)
        interior_mask[1:-1, 1:-1] = True
        phi_win_l2 = phi_l1[:, y0:y1, x0:x1].copy()
        phi_win_anchor = phi_anchor[:, y0:y1, x0:x1].copy()
        status, payload, wi = _run_worker_with_timeout(
            _bench_worker.local_l1_2d_worker,
            (phi_win_l2, phi_win_anchor, interior_mask,
             THRESHOLD, EPS_L1, L1_POLISH_MAX_ITER),
            timeout_s=L1_LOCAL_TIMEOUT_S)
        if status != 'ok':
            if status == 'timeout':
                info['n_components_timed_out'] += 1
            info['n_components_rejected'] += 1
            continue
        phi_win_new, win_info = payload
        phi_l1_candidate = phi_l1.copy()
        phi_l1_candidate[:, y0:y1, x0:x1] = phi_win_new
        T1c, T2c = _triangle_areas_2d(phi_l1_candidate[0], phi_l1_candidate[1])
        if int((T1c <= 0).sum() + (T2c <= 0).sum()) > 0:
            info['n_components_rejected'] += 1
            continue
        l1_before = float(np.abs(phi_l1[:, y0:y1, x0:x1] - phi_win_anchor).sum())
        l1_after = float(np.abs(phi_win_new - phi_win_anchor).sum())
        if l1_after >= l1_before - 1e-9:
            info['n_components_rejected'] += 1
            continue
        phi_l1 = phi_l1_candidate
        info['n_components_polished'] += 1
        info['iter_total'] += win_info['nit']
    info['wall_time'] = time.time() - t0
    return phi_l1, info


def metrics_2d(phi_now, phi_anchor):
    T1, T2 = _triangle_areas_2d(phi_now[0], phi_now[1])
    return {
        'n_neg_tri': int((T1 <= 0).sum() + (T2 <= 0).sum()),
        'min_tri': float(min(T1.min(), T2.min())),
        'L1': float(np.abs(phi_now - phi_anchor).sum()),
        'L2': float(np.linalg.norm(phi_now - phi_anchor)),
    }


def run_one_slice(phi_full, z, H, W):
    deformation = phi_full[:, z:z+1, :, :].copy()
    phi_anchor = np.stack([deformation[1, 0].copy(), deformation[2, 0].copy()])
    init = metrics_2d(phi_anchor, phi_anchor)
    t_total = time.time()
    row = {'z': int(z), 'H': int(H), 'W': int(W),
           'init_n_neg_tri': init['n_neg_tri'],
           'init_min_tri': init['min_tri']}
    if init['n_neg_tri'] == 0:
        row.update({'L2_n_neg_tri': 0, 'L2_min_tri': init['min_tri'],
                    'L2_L1': 0.0, 'L2_L2': 0.0, 'L2_t': 0.0,
                    'L2_outer_iters': 0, 'L2_local_timeouts': 0,
                    'L2_local_errors': 0, 'L2_perturb_rounds': 0,
                    'L2_final_stalled': False, 'L2_exception': None,
                    'L1_L1': 0.0, 'L1_L2': 0.0,
                    'L1_n_neg_tri': 0, 'L1_min_tri': init['min_tri'],
                    'L1_components': 0, 'L1_polished': 0,
                    'L1_rejected': 0, 'L1_skipped_large': 0, 'L1_timed_out': 0,
                    'L1_iter_total': 0, 'L1_t': 0.0,
                    'total_t': time.time() - t_total,
                    'feasible': True, 'l1_drop_pct': 0.0,
                    'phi_l1': np.stack([deformation[1, 0], deformation[2, 0]])})
        return row
    phi_l2, l2_info = run_l2_phase(deformation)
    l2_m = metrics_2d(phi_l2, phi_anchor)
    row.update({'L2_n_neg_tri': l2_m['n_neg_tri'], 'L2_min_tri': l2_m['min_tri'],
                'L2_L1': l2_m['L1'], 'L2_L2': l2_m['L2'],
                'L2_t': l2_info['wall_time'],
                'L2_outer_iters': l2_info['outer_iters_total'],
                'L2_local_timeouts': l2_info['local_timeouts_total'],
                'L2_local_errors': l2_info['local_errors_total'],
                'L2_perturb_rounds': l2_info['perturb_rounds'],
                'L2_final_stalled': bool(l2_info['final_stalled']),
                'L2_exception': l2_info['exception']})
    if l2_m['n_neg_tri'] == 0:
        phi_l1, l1_info = run_l1_polish_phase(phi_l2, phi_anchor)
        l1_m = metrics_2d(phi_l1, phi_anchor)
    else:
        phi_l1 = phi_l2
        l1_m = l2_m
        l1_info = {'n_components': 0, 'n_components_polished': 0,
                   'n_components_rejected': 0,
                   'n_components_skipped_too_large': 0,
                   'n_components_timed_out': 0,
                   'iter_total': 0, 'wall_time': 0.0}
    row.update({'L1_L1': l1_m['L1'], 'L1_L2': l1_m['L2'],
                'L1_n_neg_tri': l1_m['n_neg_tri'], 'L1_min_tri': l1_m['min_tri'],
                'L1_components': l1_info['n_components'],
                'L1_polished': l1_info['n_components_polished'],
                'L1_rejected': l1_info['n_components_rejected'],
                'L1_skipped_large': l1_info['n_components_skipped_too_large'],
                'L1_timed_out': l1_info['n_components_timed_out'],
                'L1_iter_total': l1_info['iter_total'],
                'L1_t': l1_info['wall_time'],
                'total_t': time.time() - t_total,
                'feasible': bool(l1_m['n_neg_tri'] == 0),
                'l1_drop_pct': (100.0 * (l2_m['L1'] - l1_m['L1']) / l2_m['L1']
                                 if l2_m['L1'] > 0 else 0.0),
                'phi_l1': phi_l1})
    return row


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def init_csv_if_needed():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w') as f:
            f.write(','.join(CSV_COLUMNS) + '\n')


def load_done_slices():
    if not os.path.exists(CSV_PATH):
        return set()
    try:
        df = pd.read_csv(CSV_PATH)
        return set(int(z) for z in df['z'].values)
    except Exception:
        return set()


def append_csv_row(row):
    parts = []
    for c in CSV_COLUMNS:
        v = row.get(c, '')
        if v is None:
            parts.append('')
        elif isinstance(v, float):
            parts.append(f'{v:.6g}')
        elif isinstance(v, bool):
            parts.append('True' if v else 'False')
        else:
            s = str(v).replace(',', ';').replace('\n', ' ')
            parts.append(s)
    with open(CSV_PATH, 'a') as f:
        f.write(','.join(parts) + '\n')


def save_checkpoint(arr):
    # np.savez_compressed auto-appends .npz if missing, so the tmp path
    # has to already end in .npz to land where we expect.
    tmp = CKPT_PATH + '.tmp.npz'
    np.savez_compressed(tmp, phi_corrected=arr)
    os.replace(tmp, CKPT_PATH)


def load_checkpoint(default):
    if os.path.exists(CKPT_PATH):
        try:
            with np.load(CKPT_PATH) as data:
                arr = data['phi_corrected']
                if arr.shape == default.shape:
                    return arr.copy()
        except Exception:
            pass
    return default.copy()


def log_line(msg):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def main():
    log_line(f'[start] {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}')
    log_line(f'Loading data from {DATA_PATH}')
    phi_full = np.load(DATA_PATH)
    log_line(f'shape: {phi_full.shape}')
    D, H, W = phi_full.shape[1:]
    init_csv_if_needed()
    done = load_done_slices()
    corrected = load_checkpoint(phi_full)
    log_line(f'resuming with {len(done)}/{D} slices already done')
    t_run = time.time()
    for i, z in enumerate(range(D)):
        if z in done:
            continue
        t_slice = time.time()
        try:
            row = run_one_slice(corrected, z, H, W)
        except Exception as exc:
            tb = traceback.format_exc(limit=4).replace('\n', ' | ')
            log_line(f'[ERR] z={z}  {type(exc).__name__}: {exc} :: {tb}')
            T1, T2 = _triangle_areas_2d(corrected[1, z], corrected[2, z])
            n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
            mt = float(min(T1.min(), T2.min()))
            row = {col: '' for col in CSV_COLUMNS}
            row.update({'z': z, 'H': H, 'W': W,
                        'init_n_neg_tri': n_neg, 'init_min_tri': mt,
                        'L2_n_neg_tri': -1, 'L2_min_tri': float('nan'),
                        'L2_L1': float('nan'), 'L2_L2': float('nan'),
                        'L2_t': 0.0, 'L2_outer_iters': 0,
                        'L2_local_timeouts': 0, 'L2_local_errors': 0,
                        'L2_perturb_rounds': 0, 'L2_final_stalled': True,
                        'L2_exception': f'{type(exc).__name__}: {exc}',
                        'L1_L1': float('nan'), 'L1_L2': float('nan'),
                        'L1_n_neg_tri': -1, 'L1_min_tri': float('nan'),
                        'L1_components': 0, 'L1_polished': 0,
                        'L1_rejected': 0, 'L1_skipped_large': 0,
                        'L1_timed_out': 0, 'L1_iter_total': 0,
                        'L1_t': 0.0, 'total_t': time.time() - t_slice,
                        'feasible': False, 'l1_drop_pct': 0.0,
                        'phi_l1': None})
        if row.get('phi_l1') is not None:
            corrected[1, z] = row['phi_l1'][0]
            corrected[2, z] = row['phi_l1'][1]
        row_to_write = {k: v for k, v in row.items() if k != 'phi_l1'}
        append_csv_row(row_to_write)
        done.add(z)
        n_done = len(done)
        log_line(f'[slice] z={z:>4d}  init={row["init_n_neg_tri"]:>5d}  '
                  f'L2_neg={row["L2_n_neg_tri"]:>5d}  '
                  f'L1_neg={row["L1_n_neg_tri"]:>5d}  '
                  f'L2_L1={row["L2_L1"]:>9.3f}  L1_L1={row["L1_L1"]:>9.3f}  '
                  f'drop={row["l1_drop_pct"]:>5.1f}%  '
                  f'L2_outer={row["L2_outer_iters"]:>4d} pert={row["L2_perturb_rounds"]}  '
                  f'L2_t={row["L2_t"]:>6.1f}s  L1_t={row["L1_t"]:>5.1f}s  '
                  f'feas={row["feasible"]}  done={n_done}/{D}  '
                  f'elapsed={time.time()-t_run:.0f}s')
        if n_done % CHECKPOINT_EVERY == 0:
            save_checkpoint(corrected)
            log_line(f'[ckpt] saved, done={n_done}/{D}')
    save_checkpoint(corrected)
    log_line(f'[end] {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  '
              f'done={len(done)}/{D}  total={time.time()-t_run:.0f}s')


if __name__ == '__main__':
    main()
