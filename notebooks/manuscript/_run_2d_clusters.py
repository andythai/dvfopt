"""Per-slice 2D benchmark using notebook-17/18-style full-grid SLSQP
per connected fold component.

For each z-slice:
  1. Find connected components of folded cells (any of T1/T2 <= threshold).
  2. For each component:
       - bbox + border crop
       - multi-pass full-grid L2 SLSQP on the crop (no frozen edges --
         every voxel-corner is a variable, matching notebook 17/18)
       - single L1 polish pass if L2 reached n_neg_tri = 0
  3. Splice each cluster's corrected phi back into the slice.

Records both per-slice rollups and per-cluster details so the manuscript
can characterise both. Resumable via per_slice.csv + checkpoint.npz.
"""

import os
import sys
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label, find_objects, sum_labels, binary_dilation

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from dvfopt import DEFAULT_PARAMS
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
import _bench_worker

THRESHOLD = DEFAULT_PARAMS['threshold']
ERR_TOL = DEFAULT_PARAMS['err_tol']
EPS_L1 = 1e-4

# --- Cluster geometry --------------------------------------------------
# Cluster construction:
#   1. Dilate the fold mask by MERGE_DILATION cells, so fold components
#      that are within MERGE_DILATION of each other merge into one
#      cluster (they share corners via the gap cells, so they have to
#      be solved together).
#   2. Connected-component label on the dilated mask. Each dilated
#      component is one cluster.
#   3. Bbox = bbox(dilated_component) + BBOX_PAD cells (the +1 pad
#      includes the constraint neighbours whose corners share with
#      dilated cells).
#   4. interior_mask = voxel corners adjacent to any cell in the
#      dilated component. Slice-edge corners are naturally movable
#      (no outside cell to constrain).
MERGE_DILATION = 1
BBOX_PAD = 1
MAX_CLUSTER_CELLS = 2000         # crops above this are skipped (over SLSQP cap)
MAX_CLUSTER_PER_AXIS = 60
MAX_CLUSTER_OUTER_ITERS = 20

# --- Per-cluster solver budget (notebook 17/18 style) ---------------
L2_MAX_PASSES = 15               # max L2 SLSQP passes; loop exits when n_neg=0
L2_PASS_MAX_ITER = 80
L2_PASS_TIMEOUT_S = 60
L1_POLISH_MAX_ITER = 120
L1_POLISH_TIMEOUT_S = 60

CHECKPOINT_EVERY = 5             # checkpoint every N slices

OUTPUT_DIR = os.path.join(_HERE, 'output', '2d_real_full')
os.makedirs(OUTPUT_DIR, exist_ok=True)
SLICE_CSV_PATH = os.path.join(OUTPUT_DIR, 'per_slice.csv')
CLUSTER_CSV_PATH = os.path.join(OUTPUT_DIR, 'per_cluster.csv')
CKPT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint.npz')
LOG_PATH = os.path.join(OUTPUT_DIR, 'run.log')

DATA_PATH = os.path.join(_REPO_ROOT, 'data',
                         'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')

SLICE_COLUMNS = [
    'z', 'H', 'W',
    'init_n_neg_tri', 'init_min_tri',
    'n_clusters', 'n_clusters_processed', 'n_clusters_skipped',
    'n_clusters_feasible', 'n_clusters_l2_timeout',
    'after_l2_n_neg_tri', 'after_l2_min_tri',
    'after_l1_n_neg_tri', 'after_l1_min_tri',
    'init_L1', 'after_l1_L1', 'after_l1_L2',
    'total_l2_t', 'total_l1_t', 'total_t',
    'feasible', 'l1_drop_pct',
]

CLUSTER_COLUMNS = [
    'z', 'cluster_outer_it', 'cluster_id', 'y0', 'y1', 'x0', 'x1',
    'crop_cells_y', 'crop_cells_x',
    'component_cells', 'init_n_neg_tri', 'init_min_tri',
    'after_l2_n_neg_tri', 'after_l2_min_tri',
    'l2_passes_run', 'l2_total_nit', 'l2_total_t', 'l2_any_timeout',
    'after_l1_n_neg_tri', 'after_l1_min_tri',
    'l1_nit', 'l1_t', 'l1_polished', 'l1_timed_out',
    'cluster_t', 'feasible', 'skipped_too_large',
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


def init_csvs():
    for path, cols in [(SLICE_CSV_PATH, SLICE_COLUMNS),
                       (CLUSTER_CSV_PATH, CLUSTER_COLUMNS)]:
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(','.join(cols) + '\n')


def append_row(path, cols, row):
    parts = []
    for c in cols:
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
    with open(path, 'a') as f:
        f.write(','.join(parts) + '\n')


def load_done_zs():
    if not os.path.exists(SLICE_CSV_PATH):
        return set()
    try:
        df = pd.read_csv(SLICE_CSV_PATH)
        return set(int(z) for z in df['z'].values)
    except Exception:
        return set()


def save_checkpoint(arr):
    tmp = CKPT_PATH + '.tmp.npz'
    np.savez_compressed(tmp, phi_corrected=arr)
    os.replace(tmp, CKPT_PATH)


def load_checkpoint(default):
    if not os.path.exists(CKPT_PATH):
        return default.copy()
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


def _component_corner_mask(component_mask, h, w):
    """Voxel-corner mask: a corner is movable iff at least one of its 4
    adjacent cells is in ``component_mask``. component_mask is the
    cell-grid mask shape (h, w); returned mask is voxel-corner grid
    shape (h+1, w+1).
    """
    corner_mask = np.zeros((h + 1, w + 1), dtype=bool)
    # Cell (i, j) has corners (i, j), (i+1, j), (i, j+1), (i+1, j+1).
    for di in (0, 1):
        for dj in (0, 1):
            corner_mask[di:di + h, dj:dj + w] |= component_mask
    return corner_mask


def enumerate_clusters_2d(phi_slice, max_axis, max_cells):
    """phi_slice: (2, H, W). Returns list of cluster dicts.

    Fold cells (min(T1, T2) <= 0) are dilated by ``MERGE_DILATION`` then
    connected-component labeled, so fold regions within MERGE_DILATION
    cells of each other end up in the same cluster (they share corners
    via the gap cells, so solving them independently can have them
    fight each other).
    """
    T1, T2 = _triangle_areas_2d(phi_slice[0], phi_slice[1])
    cell_min = np.minimum(T1, T2)
    cell_fold_mask = cell_min <= 0
    if not cell_fold_mask.any():
        return []
    if MERGE_DILATION > 0:
        merged_mask = binary_dilation(cell_fold_mask, iterations=MERGE_DILATION)
    else:
        merged_mask = cell_fold_mask
    labels, n_comp = cc_label(merged_mask)
    Hc, Wc = cell_fold_mask.shape
    bboxes = find_objects(labels)
    sizes = np.bincount(labels.ravel(), minlength=n_comp + 1)[1:]
    order = np.argsort(-sizes)
    clusters = []
    for idx in order:
        cid = int(idx) + 1
        bbox = bboxes[idx]
        if bbox is None:
            continue
        cy0, cy1 = bbox[0].start, bbox[0].stop
        cx0, cx1 = bbox[1].start, bbox[1].stop
        # Pad by 1 cell so the bbox includes the immediate-neighbor
        # cells (the cells whose corners share with cluster-cell corners
        # and whose constraints must stay satisfied).
        y0 = max(0, cy0 - BBOX_PAD); y1 = min(Hc, cy1 + BBOX_PAD)
        x0 = max(0, cx0 - BBOX_PAD); x1 = min(Wc, cx1 + BBOX_PAD)
        sy = y1 - y0; sx = x1 - x0
        # Movable corners = those adjacent to any cell in the (merged)
        # cluster mask. Everything else stays frozen at its current
        # value.
        cluster_in_bbox = (labels[y0:y1, x0:x1] == cid)
        interior_mask = _component_corner_mask(cluster_in_bbox, sy, sx)
        # Number of *fold* cells in this cluster (vs. dilated/merge
        # halo cells) -- this is what gets reported.
        comp_cells = int((cluster_in_bbox & cell_fold_mask[y0:y1, x0:x1]).sum())
        skipped = (sy > max_axis or sx > max_axis or sy * sx > max_cells)
        clusters.append({
            'cluster_id': cid,
            'y0': int(y0), 'y1': int(y1),
            'x0': int(x0), 'x1': int(x1),
            'crop_cells_y': sy, 'crop_cells_x': sx,
            'component_cells': comp_cells,
            'interior_mask': interior_mask,
            'n_movable_corners': int(interior_mask.sum()),
            'skipped_too_large': skipped,
        })
    return clusters


def solve_one_cluster(c, phi_slice, phi_anchor_slice, z):
    """phi_slice/phi_anchor_slice: (2, H, W). Mutates phi_slice on success."""
    y0, y1, x0, x1 = c['y0'], c['y1'], c['x0'], c['x1']
    t_cluster = time.time()
    row = {
        'z': int(z), 'cluster_id': c['cluster_id'],
        'y0': y0, 'y1': y1, 'x0': x0, 'x1': x1,
        'crop_cells_y': c['crop_cells_y'], 'crop_cells_x': c['crop_cells_x'],
        'component_cells': c['component_cells'],
        'skipped_too_large': c['skipped_too_large'],
    }
    if c['skipped_too_large']:
        row.update({'cluster_t': time.time() - t_cluster, 'feasible': False})
        return row, None
    # Voxel-corner slice (cells y0..y1 -> corners y0..y1+1)
    vy = slice(y0, y1 + 1); vx = slice(x0, x1 + 1)
    phi_crop = phi_slice[:, vy, vx].copy()
    phi_anchor_crop = phi_anchor_slice[:, vy, vx].copy()
    T1, T2 = _triangle_areas_2d(phi_crop[0], phi_crop[1])
    init_n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
    init_min_tri = float(min(T1.min(), T2.min()))
    row.update({'init_n_neg_tri': init_n_neg, 'init_min_tri': init_min_tri})
    if init_n_neg == 0:
        row.update({'after_l2_n_neg_tri': 0, 'after_l2_min_tri': init_min_tri,
                    'after_l1_n_neg_tri': 0, 'after_l1_min_tri': init_min_tri,
                    'l2_passes_run': 0, 'l2_total_nit': 0, 'l2_total_t': 0.0,
                    'l2_any_timeout': False,
                    'l1_nit': 0, 'l1_t': 0.0,
                    'l1_polished': False, 'l1_timed_out': False,
                    'cluster_t': time.time() - t_cluster, 'feasible': True})
        return row, None
    # --- Frozen-edge interior mask --------------------------------------
    # The mask is pre-computed in enumerate_clusters_2d as exactly the
    # voxel corners adjacent to a fold cell -- so SLSQP moves only what
    # needs to move and every other corner stays frozen at its current
    # (initially-valid) value.
    interior_mask = c['interior_mask']
    if not interior_mask.any():
        # Too small to have any movable interior -- skip.
        row.update({
            'after_l2_n_neg_tri': init_n_neg, 'after_l2_min_tri': init_min_tri,
            'l2_passes_run': 0, 'l2_total_nit': 0, 'l2_total_t': 0.0,
            'l2_any_timeout': False,
            'after_l1_n_neg_tri': init_n_neg, 'after_l1_min_tri': init_min_tri,
            'l1_nit': 0, 'l1_t': 0.0,
            'l1_polished': False, 'l1_timed_out': False,
            'cluster_t': time.time() - t_cluster, 'feasible': False,
        })
        return row, None

    # --- L2 multi-pass --------------------------------------------------
    phi_work = phi_crop.copy()
    l2_total_nit = 0; l2_total_t = 0.0
    l2_passes_run = 0; l2_any_timeout = False
    for p in range(L2_MAX_PASSES):
        T1, T2 = _triangle_areas_2d(phi_work[0], phi_work[1])
        if int((T1 <= 0).sum() + (T2 <= 0).sum()) == 0:
            break
        t0 = time.time()
        status, payload, info = _run_worker_with_timeout(
            _bench_worker.local_l2_2d_worker,
            (phi_work, phi_anchor_crop, interior_mask, THRESHOLD, L2_PASS_MAX_ITER),
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
    T1_l2, T2_l2 = _triangle_areas_2d(phi_work[0], phi_work[1])
    after_l2_n_neg = int((T1_l2 <= 0).sum() + (T2_l2 <= 0).sum())
    after_l2_min = float(min(T1_l2.min(), T2_l2.min()))
    # --- L1 polish (only if L2 reached n_neg = 0) -----------------------
    l1_nit = 0; l1_t = 0.0; l1_polished = False; l1_timed_out = False
    after_l1_n_neg = after_l2_n_neg
    after_l1_min = after_l2_min
    phi_l1 = phi_work
    if after_l2_n_neg == 0:
        t0 = time.time()
        status, payload, info = _run_worker_with_timeout(
            _bench_worker.local_l1_2d_worker,
            (phi_work, phi_anchor_crop, interior_mask,
             THRESHOLD, EPS_L1, L1_POLISH_MAX_ITER),
            timeout_s=L1_POLISH_TIMEOUT_S)
        l1_t = time.time() - t0
        if status == 'ok':
            phi_candidate, pass_info = payload
            T1c, T2c = _triangle_areas_2d(phi_candidate[0], phi_candidate[1])
            n_neg_c = int((T1c <= 0).sum() + (T2c <= 0).sum())
            L1_l2 = float(np.abs(phi_work - phi_anchor_crop).sum())
            L1_c = float(np.abs(phi_candidate - phi_anchor_crop).sum())
            if n_neg_c == 0 and L1_c < L1_l2 - 1e-9:
                phi_l1 = phi_candidate
                after_l1_n_neg = n_neg_c
                after_l1_min = float(min(T1c.min(), T2c.min()))
                l1_polished = True
            l1_nit = pass_info['nit']
        elif status == 'timeout':
            l1_timed_out = True
    row.update({
        'after_l2_n_neg_tri': after_l2_n_neg, 'after_l2_min_tri': after_l2_min,
        'l2_passes_run': l2_passes_run, 'l2_total_nit': l2_total_nit,
        'l2_total_t': l2_total_t, 'l2_any_timeout': l2_any_timeout,
        'after_l1_n_neg_tri': after_l1_n_neg, 'after_l1_min_tri': after_l1_min,
        'l1_nit': l1_nit, 'l1_t': l1_t,
        'l1_polished': l1_polished, 'l1_timed_out': l1_timed_out,
        'cluster_t': time.time() - t_cluster,
        'feasible': bool(after_l1_n_neg == 0),
    })
    return row, phi_l1


def process_one_slice(z, phi_full, phi_anchor_full):
    """Run all clusters in slice z. Returns (slice_row, cluster_rows, phi_slice_new).

    phi_full is mutable. If the slice has no folds, phi_slice_new is None.
    """
    H, W = phi_full.shape[2], phi_full.shape[3]
    phi_slice = np.stack([phi_full[1, z].copy(), phi_full[2, z].copy()])
    phi_anchor_slice = np.stack([phi_anchor_full[1, z], phi_anchor_full[2, z]])
    T1, T2 = _triangle_areas_2d(phi_slice[0], phi_slice[1])
    init_n_neg = int((T1 <= 0).sum() + (T2 <= 0).sum())
    init_min_tri = float(min(T1.min(), T2.min()))
    init_L1 = float(np.abs(phi_slice - phi_anchor_slice).sum())
    t_slice = time.time()
    slice_row = {
        'z': int(z), 'H': int(H), 'W': int(W),
        'init_n_neg_tri': init_n_neg, 'init_min_tri': init_min_tri,
        'init_L1': init_L1,
    }
    if init_n_neg == 0:
        slice_row.update({
            'n_clusters': 0, 'n_clusters_processed': 0,
            'n_clusters_skipped': 0, 'n_clusters_feasible': 0,
            'n_clusters_l2_timeout': 0,
            'after_l2_n_neg_tri': 0, 'after_l2_min_tri': init_min_tri,
            'after_l1_n_neg_tri': 0, 'after_l1_min_tri': init_min_tri,
            'after_l1_L1': 0.0, 'after_l1_L2': 0.0,
            'total_l2_t': 0.0, 'total_l1_t': 0.0,
            'total_t': time.time() - t_slice,
            'feasible': True, 'l1_drop_pct': 0.0,
        })
        return slice_row, [], None
    cluster_rows = []
    n_processed = 0
    n_skipped = 0
    n_feasible = 0
    n_l2_timeout = 0
    total_l2_t = 0.0
    total_l1_t = 0.0
    total_clusters = 0
    # Outer loop: each full-grid cluster solve has no frozen edges so it
    # can introduce new folds at adjacent cells. Re-find clusters and
    # re-solve until n_neg = 0 or no improvement.
    prev_n_neg = init_n_neg
    for outer_it in range(MAX_CLUSTER_OUTER_ITERS):
        clusters = enumerate_clusters_2d(phi_slice,
                                          MAX_CLUSTER_PER_AXIS,
                                          MAX_CLUSTER_CELLS)
        if not clusters:
            break
        total_clusters += len(clusters)
        any_processed_this_pass = False
        for c in clusters:
            crow, phi_l1 = solve_one_cluster(c, phi_slice, phi_anchor_slice, z)
            crow['cluster_outer_it'] = outer_it
            cluster_rows.append(crow)
            if c['skipped_too_large']:
                n_skipped += 1
                continue
            n_processed += 1
            any_processed_this_pass = True
            if crow.get('l2_any_timeout'):
                n_l2_timeout += 1
            if crow.get('feasible'):
                n_feasible += 1
            total_l2_t += crow.get('l2_total_t', 0.0)
            total_l1_t += crow.get('l1_t', 0.0)
            if phi_l1 is not None:
                y0, y1 = c['y0'], c['y1']
                x0, x1 = c['x0'], c['x1']
                vy = slice(y0, y1 + 1); vx = slice(x0, x1 + 1)
                phi_slice[:, vy, vx] = phi_l1
        # Continue iterating until n_neg=0 or MAX_CLUSTER_OUTER_ITERS.
        # Don't bail on a plateau: a non-improving iter often shuffles
        # cluster boundaries enough that the next iter clears residuals.
        T1_now, T2_now = _triangle_areas_2d(phi_slice[0], phi_slice[1])
        n_neg_now = int((T1_now <= 0).sum() + (T2_now <= 0).sum())
        if n_neg_now == 0:
            break
        if not any_processed_this_pass:
            # Nothing left to try -- all remaining clusters skipped/too-large.
            break
        prev_n_neg = n_neg_now
    T1_f, T2_f = _triangle_areas_2d(phi_slice[0], phi_slice[1])
    after_l1_n_neg = int((T1_f <= 0).sum() + (T2_f <= 0).sum())
    after_l1_min_tri = float(min(T1_f.min(), T2_f.min()))
    after_l1_L1 = float(np.abs(phi_slice - phi_anchor_slice).sum())
    after_l1_L2 = float(np.linalg.norm(phi_slice - phi_anchor_slice))
    # We don't track after_l2 separately at the slice level here; the
    # cluster CSV has per-cluster details. Slice-level we report L1
    # numbers from the final state.
    slice_row.update({
        'n_clusters': total_clusters,
        'n_clusters_processed': n_processed,
        'n_clusters_skipped': n_skipped,
        'n_clusters_feasible': n_feasible,
        'n_clusters_l2_timeout': n_l2_timeout,
        'after_l2_n_neg_tri': '',  # not separately tracked here
        'after_l2_min_tri': '',
        'after_l1_n_neg_tri': after_l1_n_neg,
        'after_l1_min_tri': after_l1_min_tri,
        'after_l1_L1': after_l1_L1, 'after_l1_L2': after_l1_L2,
        'total_l2_t': total_l2_t, 'total_l1_t': total_l1_t,
        'total_t': time.time() - t_slice,
        'feasible': bool(after_l1_n_neg == 0),
        'l1_drop_pct': (100.0 * (init_L1 - after_l1_L1) / init_L1
                         if init_L1 > 0 else 0.0),
    })
    return slice_row, cluster_rows, phi_slice


def main():
    log_line(f'[start] {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}')
    phi_full = np.load(DATA_PATH)
    log_line(f'shape: {phi_full.shape}')
    D, H, W = phi_full.shape[1:]
    init_csvs()
    done = load_done_zs()
    corrected = load_checkpoint(phi_full)
    phi_anchor = phi_full.copy()
    log_line(f'resuming with {len(done)}/{D} slices done')
    t_run = time.time()
    for z in range(D):
        if z in done:
            continue
        try:
            slice_row, cluster_rows, phi_slice_new = process_one_slice(
                z, corrected, phi_anchor)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc(limit=4).replace('\n', ' | ')
            log_line(f'[ERR] z={z}  {type(exc).__name__}: {exc} :: {tb}')
            continue
        if phi_slice_new is not None:
            corrected[1, z] = phi_slice_new[0]
            corrected[2, z] = phi_slice_new[1]
        append_row(SLICE_CSV_PATH, SLICE_COLUMNS, slice_row)
        for crow in cluster_rows:
            append_row(CLUSTER_CSV_PATH, CLUSTER_COLUMNS, crow)
        done.add(z)
        log_line(
            f'[slice] z={z:>4d}  init={slice_row["init_n_neg_tri"]:>5d}  '
            f'final={slice_row["after_l1_n_neg_tri"]:>5d}  '
            f'min_tri={slice_row["after_l1_min_tri"]:+.4f}  '
            f'L1={slice_row["after_l1_L1"]:>8.3f} '
            f'(drop {slice_row["l1_drop_pct"]:>5.1f}%)  '
            f'clusters={slice_row["n_clusters"]}('
            f'feas={slice_row["n_clusters_feasible"]} '
            f'skip={slice_row["n_clusters_skipped"]} '
            f'to={slice_row["n_clusters_l2_timeout"]})  '
            f'L2_t={slice_row["total_l2_t"]:>5.1f}s '
            f'L1_t={slice_row["total_l1_t"]:>5.1f}s  '
            f'feas={slice_row["feasible"]}  '
            f'elapsed={time.time()-t_run:.0f}s')
        if len(done) % CHECKPOINT_EVERY == 0:
            save_checkpoint(corrected)
            log_line(f'[ckpt] saved, done={len(done)}/{D}')
    save_checkpoint(corrected)
    log_line(f'[end] {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  '
              f'done={len(done)}/{D}  elapsed={time.time()-t_run:.0f}s')


if __name__ == '__main__':
    main()
