"""Headless driver for the overlapping-tile Schwarz experiment.

Mirrors the solver in overlapping_tiles_schwarz.ipynb so it can be timed
without a Jupyter kernel. Usage:  python _run_schwarz_test.py [z ...]
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
_MANU = os.path.join(_REPO, 'notebooks', 'manuscript')
sys.path.insert(0, _REPO)
sys.path.insert(0, _MANU)

import numpy as np
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d
from _bench_worker import solve_cluster_inline

THRESHOLD = 0.01
EPS_L1 = 1e-4
DATA_PATH = os.path.join(_REPO, 'data', 'corrected_correspondences_count_touching',
                         'registered_output', 'deformation3d.npy')
STUCK_Z = [30, 54, 57, 58, 61, 150, 290]

phi_full = np.load(DATA_PATH)
H, W = phi_full.shape[2], phi_full.shape[3]


def fold_stats(phi):
    T1, T2 = _triangle_areas_2d(phi[0], phi[1])
    return int((T1 <= 0).sum() + (T2 <= 0).sum()), float(min(T1.min(), T2.min()))


def slice_phi(z):
    return np.stack([phi_full[1, z].copy(), phi_full[2, z].copy()])


def make_tiles(bbox, tile, overlap):
    cy0, cy1, cx0, cx1 = bbox
    stride = max(1, tile - overlap)
    out = set()
    for y0 in range(cy0, max(cy0 + 1, cy1), stride):
        for x0 in range(cx0, max(cx0 + 1, cx1), stride):
            y1 = min(y0 + tile, H - 1)
            x1 = min(x0 + tile, W - 1)
            y0c = max(0, y1 - tile)
            x0c = max(0, x1 - tile)
            if (y1 - y0c) >= 4 and (x1 - x0c) >= 4:
                out.add((y0c, y1, x0c, x1))
    return sorted(out)


def solve_overlapping_schwarz(phi0, phi_anchor, *, tile=16, overlap=4,
                              tile_max=56, max_sweeps=25, l2_max_passes=4,
                              l2_max_iter=30, l1_max_iter=40):
    phi = phi0.copy()
    history = [dict(sweep=0, n_neg=fold_stats(phi)[0], tile=0)]
    n, m = fold_stats(phi)
    print(f'  init      : n_neg={n:5d}  min_tri={m:+.4f}', flush=True)
    cur_tile = tile
    prev_n = n
    for sweep in range(1, max_sweeps + 1):
        T1, T2 = _triangle_areas_2d(phi[0], phi[1])
        fold = np.minimum(T1, T2) <= 0
        if not fold.any():
            break
        ys, xs = np.where(fold)
        bbox = (int(ys.min()), int(ys.max()) + 1,
                int(xs.min()), int(xs.max()) + 1)
        cur_overlap = max(4, cur_tile // 4)
        tiles = make_tiles(bbox, cur_tile, cur_overlap)
        if sweep % 2 == 0:
            tiles = tiles[::-1]
        t0 = time.time()
        n_solved = 0
        for (y0, y1, x0, x1) in tiles:
            phi_win = phi[:, y0:y1 + 1, x0:x1 + 1].copy()
            t1w, t2w = _triangle_areas_2d(phi_win[0], phi_win[1])
            if not (np.minimum(t1w, t2w) <= 0).any():
                continue
            sy, sx = y1 - y0, x1 - x0
            im = np.zeros((sy + 1, sx + 1), dtype=bool)
            im[1:-1, 1:-1] = True
            c = dict(cluster_id=0, z=-1, y0=y0, y1=y1, x0=x0, x1=x1,
                     crop_cells_y=sy, crop_cells_x=sx,
                     component_cells=int((np.minimum(t1w, t2w) <= 0).sum()),
                     interior_mask=im, skipped_too_large=False)
            anc_win = phi_anchor[:, y0:y1 + 1, x0:x1 + 1].copy()
            _row, phi_l1 = solve_cluster_inline(
                c, phi_win, anc_win, THRESHOLD, EPS_L1,
                l2_max_passes, l2_max_iter, l1_max_iter)
            if phi_l1 is not None:
                yy, xx = np.where(im)
                phi[:, y0 + yy, x0 + xx] = phi_l1[:, yy, xx]
            n_solved += 1
        n, m = fold_stats(phi)
        history.append(dict(sweep=sweep, n_neg=n, tile=cur_tile))
        print(f'  sweep {sweep:2d}  : n_neg={n:5d}  min_tri={m:+.4f}  '
              f'tile={cur_tile:3d}  tiles_solved={n_solved:4d}  '
              f'({time.time()-t0:.0f}s)', flush=True)
        if n == 0:
            break
        # Plateau -> escalate tile size: a stubborn fold that a small tile
        # cannot fix gets a wider coordinated motion next sweep.
        if n >= prev_n and cur_tile < tile_max:
            cur_tile = min(tile_max, int(round(cur_tile * 1.5)))
            print(f'             plateau -> escalate tile to {cur_tile}',
                  flush=True)
        prev_n = n
    return phi, history


def main():
    targets = [int(z) for z in sys.argv[1:]] or STUCK_Z
    results = []
    for z in targets:
        phi_z = slice_phi(z)
        n0 = fold_stats(phi_z)[0]
        print(f'=== z={z}  (init n_neg={n0}) ===', flush=True)
        t0 = time.time()
        phi_c, hist = solve_overlapping_schwarz(phi_z, phi_z,
                                                tile=16, overlap=4,
                                                tile_max=56, max_sweeps=25)
        wall = time.time() - t0
        nf, mf = fold_stats(phi_c)
        results.append((z, n0, nf, mf, hist[-1]['sweep'], wall))
        print(f'    -> final n_neg={nf}  min_tri={mf:+.4f}  '
              f'sweeps={hist[-1]["sweep"]}  wall={wall:.0f}s\n', flush=True)
    print(f'{"z":>5s} {"init":>6s} {"final":>6s} {"min_tri":>9s} '
          f'{"sweeps":>7s} {"wall_s":>8s}  result', flush=True)
    for z, n0, nf, mf, sw, wall in results:
        print(f'{z:5d} {n0:6d} {nf:6d} {mf:+9.4f} {sw:7d} {wall:8.0f}  '
              f'{"CONVERGED" if nf == 0 else "still folded"}', flush=True)
    print(f'\nconverged: {sum(1 for r in results if r[2]==0)}/{len(results)}',
          flush=True)


if __name__ == '__main__':
    main()
