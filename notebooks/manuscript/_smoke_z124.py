"""Smoke test: run process_one_slice on z=124 with the new 300s timeouts.

This is the slice that got stuck in the full run -- residual clusters were
timing out at 60s on densely-coupled regions. The runner constants
L2_PASS_TIMEOUT_S / L1_POLISH_TIMEOUT_S / PER_CLUSTER_TIMEOUT_S have been
bumped to 300s; this script verifies whether the bigger budget is enough.
"""
import sys, time, os, multiprocessing as mp
sys.path.insert(0, 'notebooks/manuscript'); sys.path.insert(0, '.')
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import _run_2d_clusters as runner


def main():
    phi_full = np.load(
        'data/corrected_correspondences_count_touching/registered_output/deformation3d.npy')
    phi_anchor = phi_full.copy()
    print(f'phi shape: {phi_full.shape}', flush=True)
    print(f'runner timeouts: L2_PASS={runner.L2_PASS_TIMEOUT_S}s  '
          f'L1_POLISH={runner.L1_POLISH_TIMEOUT_S}s  '
          f'PER_CLUSTER={runner.PER_CLUSTER_TIMEOUT_S}s', flush=True)

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(
            max_workers=runner.N_PARALLEL_WORKERS,
            mp_context=ctx) as ex:
        z = 124
        t0 = time.time()
        sr, _crows, _phi = runner.process_one_slice(
            z, phi_full, phi_anchor, executor=ex)
        wall = time.time() - t0
        print(
            f'z={z}: init={sr["init_n_neg_tri"]:>4d} '
            f'after_l1_n_neg={sr["after_l1_n_neg_tri"]:>4d} '
            f'min_tri={sr["after_l1_min_tri"]:+.4f} '
            f'clusters={sr["n_clusters"]} '
            f'feas={sr["feasible"]} '
            f'wall={wall:.1f}s',
            flush=True)


if __name__ == '__main__':
    main()
