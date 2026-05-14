"""Fold-clustered 3D benchmark runner.

Pre-computes connected components of folded cells in the volume, then
runs notebook-18-style multi-pass full-grid L2 + L1 polish on one crop
per component (each crop sized to the component's bbox plus a border).
Skips the huge clean-volume search that the per-cell crop strategy
forces you to do.

Resumable via per_cluster.csv + checkpoint.npz.
"""

import os
import sys
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label
from scipy.ndimage import find_objects, sum_labels

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from dvfopt import DEFAULT_PARAMS
import _bench_worker
from _bench_worker import tet_signed_volumes

THRESHOLD = DEFAULT_PARAMS['threshold']
EPS_L1 = 1e-4

CLUSTER_BORDER = 2               # cells of free border around each cluster
MAX_CLUSTER_CELLS = 1500         # skip clusters whose crop would exceed this
                                  # (full-grid SLSQP variable count = 3 * (cells+1)^3
                                  # scaled to corners; ~3*~3500 = 10K vars max)
MAX_CLUSTER_PER_AXIS = 18        # also cap any single axis at this many cells

L2_MAX_PASSES = 6
L2_PASS_MAX_ITER = 120
L2_PASS_TIMEOUT_S = 180          # per-pass timeout; full-grid SLSQP on bigger crops is slower
L1_POLISH_MAX_ITER = 200
L1_POLISH_TIMEOUT_S = 120

CHECKPOINT_EVERY = 20

OUTPUT_DIR = os.path.join(_HERE, 'output', '3d_real_full')
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, 'per_cluster.csv')
CKPT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint.npz')
LOG_PATH = os.path.join(OUTPUT_DIR, 'run_clusters.log')

DATA_PATH = os.path.join(_REPO_ROOT, 'data',
                         'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')

CSV_COLUMNS = [
    'cluster_id', 'z0', 'z1', 'y0', 'y1', 'x0', 'x1',
    'crop_cells_z', 'crop_cells_y', 'crop_cells_x',
    'component_cells', 'component_init_n_neg_tet',
    'init_n_neg_tet', 'init_min_tet', 'init_cells_folded',
    'after_l2_n_neg_tet', 'after_l2_min_tet',
    'after_l2_L1_crop', 'after_l2_L2_crop',
    'l2_passes_run', 'l2_total_nit', 'l2_total_t', 'l2_any_timeout',
    'after_l1_n_neg_tet', 'after_l1_min_tet',
    'after_l1_L1_crop', 'after_l1_L2_crop',
    'l1_nit', 'l1_t', 'l1_timed_out', 'l1_polished',
    'crop_t', 'feasible', 'l1_drop_pct',
    'skipped_too_large',
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


def init_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w') as f:
            f.write(','.join(CSV_COLUMNS) + '\n')


def load_done_ids():
    if not os.path.exists(CSV_PATH):
        return set()
    try:
        df = pd.read_csv(CSV_PATH)
        return set(int(c) for c in df['cluster_id'].values)
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
    np.savez_compressed(tmp, phi_corrected=arr, next_cluster_idx=idx)
    os.replace(tmp, CKPT_PATH)


def load_checkpoint(default):
    if not os.path.exists(CKPT_PATH):
        return default.copy(), 0
    try:
        with np.load(CKPT_PATH) as data:
            arr = data['phi_corrected']
            idx = int(data['next_cluster_idx']) if 'next_cluster_idx' in data.files else 0
            if arr.shape == default.shape:
                return arr.copy(), idx
    except Exception:
        pass
    return default.copy(), 0


def log_line(msg):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def enumerate_clusters(phi, border, max_axis, max_cells):
    """Return list of (cluster_id, z0, z1, y0, y1, x0, x1, comp_cells, comp_neg, skipped).

    Each cluster is one connected component of cell_fold_mask, with its
    bbox padded by `border` cells and clipped to the grid. Uses
    ``scipy.ndimage.find_objects`` to get all bboxes in one pass --
    crucial when there are tens of thousands of components.
    """
    V = tet_signed_volumes(phi)
    cell_fold_mask = V.min(axis=0) <= 0
    labels, n_comp = cc_label(cell_fold_mask)
    Dc, Hc, Wc = cell_fold_mask.shape
    log_line(f'cell_fold_mask: {int(cell_fold_mask.sum())} folded cells, {n_comp} components')

    # All bboxes in one pass (returns list of n_comp slice-tuples or None).
    bboxes = find_objects(labels)
    # Component sizes in one pass.
    sizes = np.bincount(labels.ravel(), minlength=n_comp + 1)[1:]
    # Sort components by size desc -- heaviest work first so a partial
    # run still hits the most consequential clusters.
    order = np.argsort(-sizes)

    # Per-component fold-tet count: V has shape (6, Dc, Hc, Wc) so we
    # need a label volume that matches. Compute "folded tets per cell"
    # by counting (V<=0) along axis 0 (shape (Dc,Hc,Wc)), then sum that
    # over each component's label mask.
    folded_tet_per_cell = (V <= 0).sum(axis=0).astype(np.int32)
    fold_tets_per_label = sum_labels(folded_tet_per_cell, labels=labels,
                                      index=np.arange(1, n_comp + 1))

    clusters = []
    for rank, idx in enumerate(order):
        cid = int(idx) + 1   # 1-indexed component label
        bbox = bboxes[idx]   # tuple of 3 slices or None
        if bbox is None:
            continue
        z0, z1 = bbox[0].start, bbox[0].stop
        y0, y1 = bbox[1].start, bbox[1].stop
        x0, x1 = bbox[2].start, bbox[2].stop
        z0 = max(0, z0 - border); y0 = max(0, y0 - border); x0 = max(0, x0 - border)
        z1 = min(Dc, z1 + border); y1 = min(Hc, y1 + border); x1 = min(Wc, x1 + border)
        sz = z1 - z0; sy = y1 - y0; sx = x1 - x0
        comp_cells = int(sizes[idx])
        comp_neg = int(fold_tets_per_label[idx])
        skipped = sz > max_axis or sy > max_axis or sx > max_axis or sz*sy*sx > max_cells
        clusters.append((rank, int(z0), int(z1), int(y0), int(y1), int(x0), int(x1),
                          comp_cells, comp_neg, skipped))
    return clusters


def solve_one_cluster(c_tuple, phi, phi_anchor):
    cid, z0, z1, y0, y1, x0, x1, comp_cells, comp_neg, skipped = c_tuple
    sz = z1 - z0; sy = y1 - y0; sx = x1 - x0
    t_crop = time.time()
    row = {
        'cluster_id': cid, 'z0': z0, 'z1': z1, 'y0': y0, 'y1': y1,
        'x0': x0, 'x1': x1,
        'crop_cells_z': sz, 'crop_cells_y': sy, 'crop_cells_x': sx,
        'component_cells': comp_cells, 'component_init_n_neg_tet': comp_neg,
        'skipped_too_large': skipped,
    }
    if skipped:
        row.update({col: '' for col in CSV_COLUMNS
                    if col not in row and col != 'crop_t'})
        row['crop_t'] = time.time() - t_crop
        row['feasible'] = False
        return row, None
    vz = slice(z0, z1 + 1); vy = slice(y0, y1 + 1); vx = slice(x0, x1 + 1)
    phi_crop = phi[:, vz, vy, vx].copy()
    phi_anchor_crop = phi_anchor[:, vz, vy, vx].copy()
    V = tet_signed_volumes(phi_crop)
    init_n_neg = int((V <= 0).sum())
    init_min_tet = float(V.min())
    init_cells_folded = int((V.min(axis=0) <= 0).sum())
    row.update({'init_n_neg_tet': init_n_neg, 'init_min_tet': init_min_tet,
                'init_cells_folded': init_cells_folded})
    if init_n_neg == 0:
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
        return row, None
    # --- L2 multi-pass ----------------------------------------------
    phi_work = phi_crop.copy()
    l2_total_nit = 0; l2_total_t = 0.0
    l2_passes_run = 0; l2_any_timeout = False
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
    log_line(f'shape: {phi_full.shape}')
    init_csv()
    done = load_done_ids()
    corrected, _ = load_checkpoint(phi_full)
    phi_anchor = phi_full.copy()
    log_line(f'computing fold clusters on {"resumed corrected" if not np.array_equal(corrected, phi_full) else "initial"} field...')
    clusters = enumerate_clusters(corrected, CLUSTER_BORDER,
                                   MAX_CLUSTER_PER_AXIS, MAX_CLUSTER_CELLS)
    log_line(f'found {len(clusters)} clusters; '
              f'{sum(1 for c in clusters if c[-1])} too large for current cap')
    n_processable = sum(1 for c in clusters if not c[-1])
    log_line(f'will process {n_processable} clusters '
              f'(largest = {max((c[7] for c in clusters), default=0)} cells)')

    t_run = time.time()
    last_ckpt = 0
    for c_tuple in clusters:
        cid = c_tuple[0]
        if cid in done:
            continue
        try:
            row, phi_l1 = solve_one_cluster(c_tuple, corrected, phi_anchor)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc(limit=4).replace('\n', ' | ')
            log_line(f'[ERR] cid={cid}  {type(exc).__name__}: {exc} :: {tb}')
            continue
        if phi_l1 is not None:
            z0, z1, y0, y1, x0, x1 = c_tuple[1:7]
            vz = slice(z0, z1 + 1); vy = slice(y0, y1 + 1); vx = slice(x0, x1 + 1)
            corrected[:, vz, vy, vx] = phi_l1
        append_row(row)
        done.add(cid)
        log_line(f'[cluster] {cid:>5d}  '
                  f'crop={row["crop_cells_z"]}x{row["crop_cells_y"]}x{row["crop_cells_x"]}  '
                  f'comp_cells={row["component_cells"]}  '
                  f'init_neg={row.get("init_n_neg_tet", "")}  '
                  f'L2_neg={row.get("after_l2_n_neg_tet", "")}  '
                  f'L1_neg={row.get("after_l1_n_neg_tet", "")}  '
                  f'L2_passes={row.get("l2_passes_run", "")}  '
                  f'L2_t={float(row.get("l2_total_t", 0) or 0):.1f}s  '
                  f'L1_t={float(row.get("l1_t", 0) or 0):.1f}s  '
                  f'feas={row["feasible"]}  drop={float(row.get("l1_drop_pct", 0) or 0):.1f}%  '
                  f'skip={row["skipped_too_large"]}  '
                  f'elapsed={time.time()-t_run:.0f}s')
        if (cid - last_ckpt) >= CHECKPOINT_EVERY:
            save_checkpoint(corrected, cid + 1)
            last_ckpt = cid

    save_checkpoint(corrected, len(clusters))
    Vf = tet_signed_volumes(corrected)
    log_line(f'[end] {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}'
              f'  final n_neg_tet={int((Vf<=0).sum())}'
              f'  min_tet={float(Vf.min()):+.4f}'
              f'  L1={float(np.abs(corrected - phi_anchor).sum()):.2f}'
              f'  elapsed={time.time()-t_run:.0f}s')


if __name__ == '__main__':
    main()
