"""Smoke-test the per-cluster extra_dilation adaptive code on the
previously-stuck slices z=120 (6 residuals) and z=123 (1 residual),
plus the known-good baseline z=126.

Loads from checkpoint.npz so we re-attempt the SAME stuck state the
production run left behind (vs. starting from raw phi_anchor).
"""
import sys, os, time, multiprocessing as mp
sys.path.insert(0, 'notebooks/manuscript'); sys.path.insert(0, '.')
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'runner', 'notebooks/manuscript/_run_2d_clusters.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    DATA_PATH = ('data/corrected_correspondences_count_touching/'
                 'registered_output/deformation3d.npy')
    phi_full = np.load(DATA_PATH)
    phi_anchor = phi_full.copy()

    # Use the production checkpoint as the starting state so we are
    # re-attempting exactly the stuck slices the running benchmark
    # produced (rather than re-solving from raw input).
    ckpt = 'notebooks/manuscript/output/2d_real_full/checkpoint.npz'
    if os.path.exists(ckpt):
        with np.load(ckpt) as d:
            corrected = d['phi_corrected'].copy()
        print(f'[loaded checkpoint] shape={corrected.shape}', flush=True)
    else:
        corrected = phi_full.copy()
        print('[no checkpoint] starting from phi_anchor', flush=True)

    targets = [120, 123, 126]
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as ex:
        for z in targets:
            t0 = time.time()
            sr, _, phi_new = runner.process_one_slice(z, corrected, phi_anchor, executor=ex)
            wall = time.time() - t0
            print(f'z={z:>3d}: '
                  f'init={sr["init_n_neg_tri"]:>4d}  '
                  f'final={sr["after_l1_n_neg_tri"]:>4d}  '
                  f'min_tri={sr["after_l1_min_tri"]:+.4f}  '
                  f'feas={sr["feasible"]}  '
                  f'clusters={sr["n_clusters"]}('
                  f'feas={sr["n_clusters_feasible"]} '
                  f'to={sr["n_clusters_l2_timeout"]})  '
                  f'wall={wall:.1f}s', flush=True)
            if phi_new is not None:
                corrected[1, z] = phi_new[0]
                corrected[2, z] = phi_new[1]


if __name__ == '__main__':
    main()
