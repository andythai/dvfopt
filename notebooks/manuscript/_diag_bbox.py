"""Diagnose hypothesis #1: residual clusters are stuck because the bbox
is too small. Re-run z=126 with progressively larger BBOX_PAD and see
how the residual count changes.
"""
import sys, time, os, multiprocessing as mp
sys.path.insert(0, 'notebooks/manuscript'); sys.path.insert(0, '.')
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import importlib.util


def load_runner(bbox_pad, merge_dilation):
    """Monkey-patch runner module's BBOX_PAD / MERGE_DILATION constants
    before exec, so we can sweep without editing the file."""
    spec = importlib.util.spec_from_file_location(
        'runner_diag', 'notebooks/manuscript/_run_2d_clusters.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.BBOX_PAD = bbox_pad
    runner.MERGE_DILATION = merge_dilation
    return runner


def main():
    phi_full = np.load(
        'data/corrected_correspondences_count_touching/registered_output/deformation3d.npy')
    phi_anchor = phi_full.copy()

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as ex:
        # BBOX_PAD>1 confirmed harmful (over-constrains the QP). Sweep
        # only MERGE_DILATION to find the residual minimum.
        for bbox_pad in [1]:
            for merge in [2, 3, 4, 5]:
                runner = load_runner(bbox_pad, merge)
                corrected = phi_full.copy()
                t0 = time.time()
                sr, _, _ = runner.process_one_slice(126, corrected, phi_anchor, executor=ex)
                wall = time.time() - t0
                print(f'BBOX_PAD={bbox_pad} MERGE={merge}: '
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
