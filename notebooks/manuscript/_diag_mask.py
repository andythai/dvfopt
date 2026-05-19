"""Compare interior_mask strategies:
  A. component-shaped (current default): corners adjacent to cluster cells
  B. rectangular: all corners inside bbox boundary (matches earlier smoke)
"""
import sys, time, os, multiprocessing as mp
sys.path.insert(0, 'notebooks/manuscript'); sys.path.insert(0, '.')
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import importlib.util


def load_runner(mask_strategy, merge):
    spec = importlib.util.spec_from_file_location(
        f'runner_{mask_strategy}_{merge}', 'notebooks/manuscript/_run_2d_clusters.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.MERGE_DILATION = merge
    runner.BBOX_PAD = 1

    if mask_strategy == 'rect':
        # Override enumerate_clusters_2d to use rectangular interior_mask
        # like the earlier working version.
        from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
        from scipy.ndimage import (label as cc_label, find_objects, binary_dilation)

        def enumerate_rect(phi_slice, max_axis, max_cells):
            T1, T2 = _triangle_areas_2d(phi_slice[0], phi_slice[1])
            cell_min = np.minimum(T1, T2)
            cell_fold_mask = cell_min <= 0
            if not cell_fold_mask.any():
                return []
            if runner.MERGE_DILATION > 0:
                merged_mask = binary_dilation(cell_fold_mask, iterations=runner.MERGE_DILATION)
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
                y0 = max(0, cy0 - runner.BBOX_PAD); y1 = min(Hc, cy1 + runner.BBOX_PAD)
                x0 = max(0, cx0 - runner.BBOX_PAD); x1 = min(Wc, cx1 + runner.BBOX_PAD)
                sy = y1 - y0; sx = x1 - x0
                # RECTANGULAR interior_mask: every corner inside the
                # bbox boundary is movable (vs the component-shaped
                # version which only marks fold-adjacent corners).
                interior_mask = np.zeros((sy + 1, sx + 1), dtype=bool)
                if sy + 1 > 2 and sx + 1 > 2:
                    interior_mask[1:-1, 1:-1] = True
                cluster_in_bbox = (labels[y0:y1, x0:x1] == cid)
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

        runner.enumerate_clusters_2d = enumerate_rect
    return runner


def main():
    phi_full = np.load(
        'data/corrected_correspondences_count_touching/registered_output/deformation3d.npy')
    phi_anchor = phi_full.copy()
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as ex:
        for mask in ['component', 'rect']:
            for merge in [1, 2]:
                runner = load_runner(mask, merge)
                corrected = phi_full.copy()
                t0 = time.time()
                sr, _, _ = runner.process_one_slice(126, corrected, phi_anchor, executor=ex)
                wall = time.time() - t0
                print(f'mask={mask:<10s} MERGE={merge}: '
                      f'init={sr["init_n_neg_tri"]:>4d} '
                      f'final={sr["after_l1_n_neg_tri"]:>4d} '
                      f'min_tri={sr["after_l1_min_tri"]:+.4f} '
                      f'clusters={sr["n_clusters"]:>4d} '
                      f'feas_clusters={sr["n_clusters_feasible"]:>4d} '
                      f'feas={sr["feasible"]} '
                      f'wall={wall:.1f}s',
                      flush=True)


if __name__ == '__main__':
    main()
