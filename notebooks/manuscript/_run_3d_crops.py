"""Crop-based 3D benchmark runner.

Splits the volume into overlapping cell-crops, then runs the full-grid
multi-pass L2 + L1 polish solver (notebook 17/18 style) on each crop in
sequence. Each per-crop SLSQP call runs in a child process with a
wall-clock timeout. Resumable via per-crop CSV + checkpoint.

This bypasses the windowed-SLSQP boundary-freeze limitation that
prevented the per-window solver from converging on the full volume:
each crop's SLSQP can move its boundary voxels freely, and the overlap
between adjacent crops smooths corrections across crop seams.
"""

import os
import sys
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from dvfopt import DEFAULT_PARAMS
import _bench_worker
from _bench_worker import tet_signed_volumes

THRESHOLD = DEFAULT_PARAMS['threshold']
EPS_L1 = 1e-4

# --- Crop geometry ---------------------------------------------------
# Cells per axis. Voxel-corner crop is (CROP_Z+1) x (CROP_Y+1) x (CROP_X+1).
# Variables per crop = 3 * (CROP_Z+1)*(CROP_Y+1)*(CROP_X+1).
# Notebook 18 ran full-grid SLSQP on 5x10x10 (1500 vars) in 100-450s.
# 8^3 = 729 corners -> 2187 vars. Borderline; tune down if too slow.
CROP_Z = 7
CROP_Y = 7
CROP_X = 7
OVERLAP = 1     # cells of overlap between adjacent crops

# --- Per-crop solver budget ------------------------------------------
L2_MAX_PASSES = 6                # like notebook 17's multi-pass loop
L2_PASS_MAX_ITER = 120           # SLSQP iter cap per pass
L2_PASS_TIMEOUT_S = 120          # wall-clock timeout per pass
L1_POLISH_MAX_ITER = 200
L1_POLISH_TIMEOUT_S = 90
SKIP_CROP_IF_NO_FOLDS = True     # crops with all V_t >= threshold are skipped

# Save a checkpoint of the corrected volume every N processed crops.
CHECKPOINT_EVERY = 25

OUTPUT_DIR = os.path.join(_HERE, 'output', '3d_real_full')
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, 'per_crop.csv')
CKPT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint.npz')
LOG_PATH = os.path.join(OUTPUT_DIR, 'run.log')

DATA_PATH = os.path.join(_REPO_ROOT, 'data',
                         'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')


CSV_COLUMNS = [
    'crop_id', 'z0', 'z1', 'y0', 'y1', 'x0', 'x1',
    'init_n_neg_tet', 'init_min_tet', 'init_cells_folded',
    'after_l2_n_neg_tet', 'after_l2_min_tet',
    'after_l2_L1_crop', 'after_l2_L2_crop',
    'l2_passes_run', 'l2_total_nit', 'l2_total_t', 'l2_any_timeout',
    'after_l1_n_neg_tet', 'after_l1_min_tet',
    'after_l1_L1_crop', 'after_l1_L2_crop',
    'l1_nit', 'l1_t', 'l1_timed_out', 'l1_polished',
    'crop_t', 'feasible', 'l1_drop_pct',
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


def enumerate_crops(D, H, W, crop_z, crop_y, crop_x, overlap):
    """Generate (z0, z1, y0, y1, x0, x1) cell-bbox tuples covering the
    whole volume with the given overlap. Last crop along each axis is
    flush with the field edge (may be smaller than `crop_*` if not a
    multiple).
    """
    step_z = max(1, crop_z - overlap)
    step_y = max(1, crop_y - overlap)
    step_x = max(1, crop_x - overlap)

    def _axis(n_cells, crop, step):
        starts = list(range(0, max(1, n_cells - crop + 1), step))
        if not starts or starts[-1] + crop < n_cells:
            starts.append(max(0, n_cells - crop))
        # Deduplicate
        seen = []
        for s in starts:
            if s not in seen:
                seen.append(s)
        return seen

    zs = _axis(D, crop_z, step_z)
    ys = _axis(H, crop_y, step_y)
    xs = _axis(W, crop_x, step_x)
    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                z1 = min(D, z0 + crop_z)
                y1 = min(H, y0 + crop_y)
                x1 = min(W, x0 + crop_x)
                yield int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def init_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w') as f:
            f.write(','.join(CSV_COLUMNS) + '\n')


def load_done_ids():
    if not os.path.exists(CSV_PATH):
        return set()
    try:
        df = pd.read_csv(CSV_PATH)
        return set(int(c) for c in df['crop_id'].values)
    except Exception:
        return set()


def append_row(row):
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
            parts.append(str(v))
    with open(CSV_PATH, 'a') as f:
        f.write(','.join(parts) + '\n')


def save_checkpoint(arr, idx):
    tmp = CKPT_PATH + '.tmp'
    np.savez_compressed(tmp, phi_corrected=arr, next_crop_idx=idx)
    os.replace(tmp, CKPT_PATH)


def load_checkpoint(default):
    if not os.path.exists(CKPT_PATH):
        return default.copy(), 0
    try:
        with np.load(CKPT_PATH) as data:
            arr = data['phi_corrected']
            idx = int(data['next_crop_idx']) if 'next_crop_idx' in data.files else 0
            if arr.shape == default.shape:
                return arr.copy(), idx
    except Exception:
        pass
    return default.copy(), 0


def log_line(msg):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def solve_one_crop(crop_id, z0, z1, y0, y1, x0, x1, phi, phi_anchor):
    """Run multi-pass L2 + L1 polish on one crop. Mutates ``phi`` in place
    on the crop slice. Returns the row dict.
    """
    vz = slice(z0, z1 + 1); vy = slice(y0, y1 + 1); vx = slice(x0, x1 + 1)
    phi_crop = phi[:, vz, vy, vx].copy()
    phi_anchor_crop = phi_anchor[:, vz, vy, vx].copy()
    V = tet_signed_volumes(phi_crop)
    init_n_neg = int((V <= 0).sum())
    init_min_tet = float(V.min())
    init_cells_folded = int((V.min(axis=0) <= 0).sum())
    row = {
        'crop_id': crop_id, 'z0': z0, 'z1': z1, 'y0': y0, 'y1': y1,
        'x0': x0, 'x1': x1,
        'init_n_neg_tet': init_n_neg, 'init_min_tet': init_min_tet,
        'init_cells_folded': init_cells_folded,
    }
    t_crop = time.time()
    if init_n_neg == 0 and SKIP_CROP_IF_NO_FOLDS:
        row.update({
            'after_l2_n_neg_tet': 0, 'after_l2_min_tet': init_min_tet,
            'after_l2_L1_crop': 0.0, 'after_l2_L2_crop': 0.0,
            'l2_passes_run': 0, 'l2_total_nit': 0, 'l2_total_t': 0.0,
            'l2_any_timeout': False,
            'after_l1_n_neg_tet': 0, 'after_l1_min_tet': init_min_tet,
            'after_l1_L1_crop': 0.0, 'after_l1_L2_crop': 0.0,
            'l1_nit': 0, 'l1_t': 0.0, 'l1_timed_out': False,
            'l1_polished': False,
            'crop_t': time.time() - t_crop, 'feasible': True,
            'l1_drop_pct': 0.0,
        })
        return row, None  # no change to phi

    # --- L2 multi-pass ----------------------------------------------
    phi_work = phi_crop.copy()
    l2_total_nit = 0
    l2_total_t = 0.0
    l2_passes_run = 0
    l2_any_timeout = False
    for p in range(L2_MAX_PASSES):
        V = tet_signed_volumes(phi_work)
        if int((V <= 0).sum()) == 0:
            break
        t0 = time.time()
        status, payload, info = _run_worker_with_timeout(
            _bench_worker.full_grid_l2_3d_worker,
            (phi_work, phi_anchor_crop, THRESHOLD, L2_PASS_MAX_ITER),
            timeout_s=L2_PASS_TIMEOUT_S)
        elapsed = time.time() - t0
        l2_total_t += elapsed
        l2_passes_run += 1
        if status == 'ok':
            phi_new, pass_info = payload
            phi_work = phi_new
            l2_total_nit += pass_info['nit']
        elif status == 'timeout':
            l2_any_timeout = True
            break
        else:
            break
    V_after_l2 = tet_signed_volumes(phi_work)
    after_l2_n_neg = int((V_after_l2 <= 0).sum())
    after_l2_min = float(V_after_l2.min())
    after_l2_L1 = float(np.abs(phi_work - phi_anchor_crop).sum())
    after_l2_L2 = float(np.linalg.norm(phi_work - phi_anchor_crop))

    # --- L1 polish (only if L2 reached feasibility) -----------------
    l1_nit = 0; l1_t = 0.0; l1_timed_out = False; l1_polished = False
    after_l1_n_neg = after_l2_n_neg; after_l1_min = after_l2_min
    after_l1_L1 = after_l2_L1; after_l1_L2 = after_l2_L2
    phi_l1 = phi_work
    if after_l2_n_neg == 0:
        t0 = time.time()
        status, payload, info = _run_worker_with_timeout(
            _bench_worker.full_grid_l1_3d_worker,
            (phi_work, phi_anchor_crop, THRESHOLD, EPS_L1, L1_POLISH_MAX_ITER),
            timeout_s=L1_POLISH_TIMEOUT_S)
        l1_t = time.time() - t0
        if status == 'ok':
            phi_candidate, pass_info = payload
            Vc = tet_signed_volumes(phi_candidate)
            n_neg_c = int((Vc <= 0).sum())
            L1_c = float(np.abs(phi_candidate - phi_anchor_crop).sum())
            if n_neg_c == 0 and L1_c < after_l2_L1 - 1e-9:
                phi_l1 = phi_candidate
                after_l1_n_neg = n_neg_c
                after_l1_min = float(Vc.min())
                after_l1_L1 = L1_c
                after_l1_L2 = float(np.linalg.norm(phi_l1 - phi_anchor_crop))
                l1_polished = True
            l1_nit = pass_info['nit']
        elif status == 'timeout':
            l1_timed_out = True

    row.update({
        'after_l2_n_neg_tet': after_l2_n_neg, 'after_l2_min_tet': after_l2_min,
        'after_l2_L1_crop': after_l2_L1, 'after_l2_L2_crop': after_l2_L2,
        'l2_passes_run': l2_passes_run, 'l2_total_nit': l2_total_nit,
        'l2_total_t': l2_total_t, 'l2_any_timeout': l2_any_timeout,
        'after_l1_n_neg_tet': after_l1_n_neg, 'after_l1_min_tet': after_l1_min,
        'after_l1_L1_crop': after_l1_L1, 'after_l1_L2_crop': after_l1_L2,
        'l1_nit': l1_nit, 'l1_t': l1_t,
        'l1_timed_out': l1_timed_out, 'l1_polished': l1_polished,
        'crop_t': time.time() - t_crop,
        'feasible': bool(after_l1_n_neg == 0),
        'l1_drop_pct': (100.0 * (after_l2_L1 - after_l1_L1) / after_l2_L1
                         if after_l2_L1 > 0 else 0.0),
    })
    return row, phi_l1


def main():
    log_line(f'[start] {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}')
    phi_full = np.load(DATA_PATH)
    log_line(f'shape: {phi_full.shape}  crop=(z={CROP_Z},y={CROP_Y},x={CROP_X}) overlap={OVERLAP}')
    D, H, W = phi_full.shape[1:]
    # Cell-grid dims are (D-1, H-1, W-1) for a (D,H,W) voxel-corner field.
    crops = list(enumerate_crops(D - 1, H - 1, W - 1,
                                  CROP_Z, CROP_Y, CROP_X, OVERLAP))
    log_line(f'total crops: {len(crops)}')
    init_csv()
    done = load_done_ids()
    corrected, resume_idx = load_checkpoint(phi_full)
    phi_anchor = phi_full.copy()
    log_line(f'resuming with {len(done)} crops done, next_idx={resume_idx}')

    t_run = time.time()
    last_ckpt_idx = 0
    for cid, (z0, z1, y0, y1, x0, x1) in enumerate(crops):
        if cid in done:
            continue
        try:
            row, phi_l1 = solve_one_crop(cid, z0, z1, y0, y1, x0, x1,
                                          corrected, phi_anchor)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc(limit=4).replace('\n', ' | ')
            log_line(f'[ERR] cid={cid}  {type(exc).__name__}: {exc} :: {tb}')
            continue
        if phi_l1 is not None:
            vz = slice(z0, z1 + 1); vy = slice(y0, y1 + 1); vx = slice(x0, x1 + 1)
            corrected[:, vz, vy, vx] = phi_l1
        append_row(row)
        done.add(cid)
        log_line(f'[crop] {cid:>5d}/{len(crops)}  '
                  f'bbox=z[{z0:>3d},{z1:>3d}) y[{y0:>3d},{y1:>3d}) x[{x0:>3d},{x1:>3d})  '
                  f'init_neg={row["init_n_neg_tet"]:>4d}  '
                  f'L2_neg={row["after_l2_n_neg_tet"]:>4d}  '
                  f'L1_neg={row["after_l1_n_neg_tet"]:>4d}  '
                  f'L2_passes={row["l2_passes_run"]}  L2_t={row["l2_total_t"]:>5.1f}s  '
                  f'L1_t={row["l1_t"]:>5.1f}s  feas={row["feasible"]}  '
                  f'drop={row["l1_drop_pct"]:>5.1f}%  '
                  f'elapsed={time.time()-t_run:.0f}s')
        if (cid - last_ckpt_idx) >= CHECKPOINT_EVERY:
            save_checkpoint(corrected, cid + 1)
            last_ckpt_idx = cid
            log_line(f'[ckpt] saved at cid={cid+1}')

    save_checkpoint(corrected, len(crops))
    Vf = tet_signed_volumes(corrected)
    log_line(f'[end] {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}'
              f'  final n_neg_tet={int((Vf<=0).sum())}'
              f'  min_tet={float(Vf.min()):+.4f}'
              f'  L1={float(np.abs(corrected - phi_anchor).sum()):.2f}'
              f'  elapsed={time.time()-t_run:.0f}s')


if __name__ == '__main__':
    main()
