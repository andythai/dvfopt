"""
Parallelized iterative SLSQP optimisation for correcting negative Jacobian
determinants in 2D deformation (displacement) fields.

Hybrid approach:

* When multiple non-overlapping negative-Jdet windows exist, they are
  dispatched to a ``ProcessPoolExecutor`` in parallel.
* When only one window can be selected (dense / overlapping negatives),
  the optimisation runs **directly in-process** with full serial-style
  adaptive window growth — avoiding the massive Windows ``spawn``
  overhead that would otherwise dominate.

Usage::

    from modules.dvfopt_parallel import iterative_parallel

"""

import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


# ---------------------------------------------------------------------------
# Shared helpers from dvfopt
# ---------------------------------------------------------------------------
from modules.dvfopt import (
    DEFAULT_PARAMS,
    _log,
    objectiveEuc,
    jacobian_det2D,
    jacobian_constraint,
    nearest_center,
    get_nearest_center,
    argmin_excluding_edges,
    get_phi_sub_flat,
)


# ---------------------------------------------------------------------------
# Standalone worker (must be picklable)
# ---------------------------------------------------------------------------
def _optimize_single_window(
    phi_sub_flat,
    phi_init_sub_flat,
    submatrix_size,
    is_at_edge,
    window_reached_max,
    threshold,
    max_minimize_iter,
    method_name,
):
    """Run SLSQP on one sub-window.  Returns ``(result_x, elapsed)``."""

    if window_reached_max:
        nlc = NonlinearConstraint(
            lambda phi1: jacobian_constraint(phi1, submatrix_size, False),
            threshold, np.inf,
        )
        constraints = [nlc]
    else:
        nlc = NonlinearConstraint(
            lambda phi1: jacobian_constraint(phi1, submatrix_size, not is_at_edge),
            threshold, np.inf,
        )

        if not is_at_edge:
            edge_mask = np.zeros((submatrix_size, submatrix_size), dtype=bool)
            edge_mask[[0, -1], :] = True
            edge_mask[:, [0, -1]] = True

            edge_indices = np.argwhere(edge_mask)
            fixed_indices = []
            y_offset_sub = submatrix_size * submatrix_size
            for y, x in edge_indices:
                idx = y * submatrix_size + x
                fixed_indices.extend([idx, idx + y_offset_sub])

            fixed_values = phi_sub_flat[fixed_indices]
            A_eq = np.zeros((len(fixed_indices), phi_sub_flat.size))
            for row, idx in enumerate(fixed_indices):
                A_eq[row, idx] = 1
            linear_constraint = LinearConstraint(A_eq, fixed_values, fixed_values)
            constraints = [nlc, linear_constraint]
        else:
            constraints = [nlc]

    t0 = time.time()
    result = minimize(
        lambda phi1: objectiveEuc(phi1, phi_init_sub_flat),
        phi_sub_flat,
        constraints=constraints,
        options={"maxiter": max_minimize_iter, "disp": False},
        method=method_name,
    )
    elapsed = time.time() - t0
    return result.x, elapsed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_negative_pixels(jacobian_matrix, threshold, err_tol):
    """Return list of (y, x) for inner pixels below threshold, worst first."""
    inner = jacobian_matrix[0, 1:-1, 1:-1]
    ys, xs = np.where(inner <= threshold - err_tol)
    vals = inner[ys, xs]
    order = np.argsort(vals)
    return [(int(ys[i]) + 1, int(xs[i]) + 1) for i in order]


def _window_bounds(cy, cx, d):
    return (cy - d, cy + d, cx - d, cx + d)


def _windows_overlap(b1, b2):
    return not (b1[1] < b2[0] or b2[1] < b1[0] or b1[3] < b2[2] or b2[3] < b1[2])


def _select_non_overlapping(neg_pixels, pixel_window_sizes, slice_shape,
                             near_cent_dict):
    """Greedily select non-overlapping windows (each pixel has its own size)."""
    selected = []
    used_bounds = []

    for neg_idx in neg_pixels:
        ws = pixel_window_sizes[neg_idx]
        d = ws // 2
        cz, cy, cx = get_nearest_center(neg_idx, slice_shape, ws, near_cent_dict)
        bounds = _window_bounds(cy, cx, d)

        overlaps = False
        for ub in used_bounds:
            if _windows_overlap(bounds, ub):
                overlaps = True
                break

        if not overlaps:
            selected.append((neg_idx, (cz, cy, cx), ws))
            used_bounds.append(bounds)

    return selected


def _apply_result(phi, result_x, cy, cx, sub_size):
    """Write optimised sub-window back into *phi*."""
    d = sub_size // 2
    pixels = sub_size * sub_size
    phi[1, cy - d:cy + d + 1, cx - d:cx + d + 1] = \
        result_x[:pixels].reshape(sub_size, sub_size)
    phi[0, cy - d:cy + d + 1, cx - d:cx + d + 1] = \
        result_x[pixels:].reshape(sub_size, sub_size)


def _edge_flags(cy, cx, d, slice_shape, max_window):
    """Return (is_at_edge, window_reached_max) for a window."""
    start_y, end_y = cy - d, cy + d
    start_x, end_x = cx - d, cx + d
    max_y, max_x = slice_shape[1:]
    is_at_edge = (start_y == 0 or end_y >= max_y - 1
                  or start_x == 0 or end_x >= max_x - 1)
    window_reached_max = (d * 2 + 1) >= max_window
    return is_at_edge, window_reached_max


# ---------------------------------------------------------------------------
# Serial inner loop — runs when batch_size == 1
# ---------------------------------------------------------------------------
def _serial_fix_pixel(
    neg_index_tuple, phi, phi_init, jacobian_matrix,
    slice_shape, near_cent_dict, window_counts,
    starting_window_size, max_per_index_iter, max_minimize_iter,
    max_window, threshold, err_tol, methodName, verbose,
    error_list, num_neg_jac, min_jdet_list, iter_times,
):
    """Fix a single pixel using the serial adaptive-window inner loop.

    Mirrors the inner ``while`` loop of ``iterative_with_jacobians2``
    exactly: grow the window by 2 each sub-iteration until the local
    region is clean or the window hits the grid boundary.

    Mutates *phi*, *jacobian_matrix*, and the accumulator lists in-place,
    returns the updated jacobian_matrix.
    """
    submatrix_size = starting_window_size
    per_index_iter = 0
    window_reached_max = False

    while (
        submatrix_size == starting_window_size
        or (
            (not window_reached_max)
            and per_index_iter < max_per_index_iter
            and (jacobian_matrix[0,
                    cy - center_distance:cy + center_distance + 1,
                    cx - center_distance:cx + center_distance + 1]
                 < threshold - err_tol).any()
        )
    ):
        per_index_iter += 1

        if submatrix_size < max_window:
            submatrix_size += 2
        else:
            window_reached_max = True

        window_counts[submatrix_size] += 1

        cz, cy, cx = get_nearest_center(
            neg_index_tuple, slice_shape, submatrix_size, near_cent_dict)
        d = submatrix_size // 2
        center_distance = d

        phi_init_sub_flat = get_phi_sub_flat(phi_init, cz, cy, cx, slice_shape, d)
        phi_sub_flat = get_phi_sub_flat(phi, cz, cy, cx, slice_shape, d)

        is_at_edge, w_max = _edge_flags(cy, cx, d, slice_shape, max_window)
        window_reached_max = window_reached_max or w_max

        # Run optimisation directly — no process pool
        result_x, elapsed = _optimize_single_window(
            phi_sub_flat, phi_init_sub_flat, submatrix_size,
            is_at_edge, window_reached_max,
            threshold, max_minimize_iter, methodName,
        )
        iter_times.append(elapsed)

        _apply_result(phi, result_x, cy, cx, submatrix_size)

        jacobian_matrix = jacobian_det2D(phi)
        cur_neg = int((jacobian_matrix <= 0).sum())
        cur_min = float(jacobian_matrix.min())
        num_neg_jac.append(cur_neg)
        min_jdet_list.append(cur_min)
        error_list.append(np.sqrt(np.sum((phi - phi_init) ** 2)))

        if cur_min > threshold - err_tol:
            break

    return jacobian_matrix, submatrix_size, per_index_iter


# ---------------------------------------------------------------------------
# Main hybrid algorithm
# ---------------------------------------------------------------------------
def iterative_parallel(
    deformation_i,
    methodName="SLSQP",
    verbose=1,
    save_path=None,
    plot_every=0,
    threshold=None,
    err_tol=None,
    max_iterations=None,
    max_per_index_iter=None,
    max_minimize_iter=None,
    starting_window_size=None,
    max_workers=None,
):
    """Hybrid serial/parallel iterative SLSQP correction.

    * batch_size > 1 → ``ProcessPoolExecutor`` (true parallelism)
    * batch_size == 1 → direct in-process with full adaptive window
      growth (avoids Windows ``spawn`` overhead)

    Parameters
    ----------
    max_workers : int or None
        Number of worker processes.  ``None`` → ``os.cpu_count()``.

    Returns
    -------
    phi : ndarray, shape ``(2, H, W)``
    """
    # Resolve parameters
    p = dict(DEFAULT_PARAMS)
    for name, val in [
        ("threshold", threshold),
        ("err_tol", err_tol),
        ("max_iterations", max_iterations),
        ("max_per_index_iter", max_per_index_iter),
        ("max_minimize_iter", max_minimize_iter),
        ("starting_window_size", starting_window_size),
    ]:
        if val is not None:
            p[name] = val
    threshold = p["threshold"]
    err_tol = p["err_tol"]
    max_iterations = p["max_iterations"]
    max_per_index_iter = p["max_per_index_iter"]
    max_minimize_iter = p["max_minimize_iter"]
    starting_window_size = p["starting_window_size"]

    if verbose is True:
        verbose = 1
    elif verbose is False:
        verbose = 0

    if max_workers is None:
        max_workers = os.cpu_count()

    # Accumulators
    error_list = []
    num_neg_jac = []
    iter_times = []
    min_jdet_list = []
    window_counts = defaultdict(int)

    start_time = time.time()
    H, W = deformation_i.shape[-2:]
    slice_shape = (1, H, W)
    max_window = min(H, W) - 1
    near_cent_dict = {}

    phi = np.zeros((2, H, W))
    phi[1] = deformation_i[-1]
    phi[0] = deformation_i[-2]
    phi_init = phi.copy()

    _log(verbose, 1,
         f"[init] Grid {H}x{W}  |  threshold={threshold}  "
         f"|  method={methodName}  |  workers={max_workers}")

    jacobian_matrix = jacobian_det2D(phi)
    init_neg = int((jacobian_matrix <= 0).sum())
    init_min = float(jacobian_matrix.min())
    min_jdet_list.append(init_min)
    num_neg_jac.append(init_neg)

    _log(verbose, 1,
         f"[init] Neg-Jdet pixels: {init_neg}  |  min Jdet: {init_min:.6f}")

    # Per-pixel window size tracker for parallel batches
    pixel_window_sizes = {}

    iteration = 0
    serial_iters = 0
    parallel_iters = 0
    executor = None  # lazy — only created if we actually need parallelism

    try:
        while (iteration < max_iterations
               and (jacobian_matrix[0, 1:-1, 1:-1] <= threshold - err_tol).any()):
            iteration += 1

            neg_pixels = _find_negative_pixels(jacobian_matrix, threshold, err_tol)
            if not neg_pixels:
                break

            # Assign / grow window sizes for batching decision
            current_neg_set = set(neg_pixels)
            pixel_window_sizes = {
                k: v for k, v in pixel_window_sizes.items()
                if k in current_neg_set
            }
            for px in neg_pixels:
                if px in pixel_window_sizes:
                    if pixel_window_sizes[px] < max_window:
                        pixel_window_sizes[px] += 2
                else:
                    pixel_window_sizes[px] = starting_window_size + 2
                pixel_window_sizes[px] = min(pixel_window_sizes[px], max_window)

            # Select non-overlapping batch
            batch = _select_non_overlapping(
                neg_pixels, pixel_window_sizes, slice_shape, near_cent_dict
            )

            if len(batch) <= 1:
                # ──────────────────────────────────────────────────────
                # SERIAL PATH — run directly, no process pool overhead
                # ──────────────────────────────────────────────────────
                serial_iters += 1
                neg_idx = neg_pixels[0]

                _log(verbose, 1,
                     f"[iter {iteration:4d}]  serial  "
                     f"fix ({neg_idx[0]:3d},{neg_idx[1]:3d})  "
                     f"neg_pixels={len(neg_pixels)}")

                jacobian_matrix, sub_size, sub_iters = _serial_fix_pixel(
                    neg_idx, phi, phi_init, jacobian_matrix,
                    slice_shape, near_cent_dict, window_counts,
                    starting_window_size, max_per_index_iter,
                    max_minimize_iter, max_window,
                    threshold, err_tol, methodName, verbose,
                    error_list, num_neg_jac, min_jdet_list, iter_times,
                )

                cur_neg = int((jacobian_matrix <= 0).sum())
                cur_min = float(jacobian_matrix.min())
                cur_err = error_list[-1] if error_list else 0.0
                _log(verbose, 1,
                     f"         -> neg_jdet {cur_neg:5d}  "
                     f"min_jdet {cur_min:+.6f}  L2 {cur_err:.4f}  "
                     f"win {sub_size}  sub-iters {sub_iters}")

            else:
                # ──────────────────────────────────────────────────────
                # PARALLEL PATH — submit batch to process pool
                # ──────────────────────────────────────────────────────
                parallel_iters += 1

                # Lazy-create executor on first parallel batch
                if executor is None:
                    executor = ProcessPoolExecutor(max_workers=max_workers)

                batch_sizes = [ws for _, _, ws in batch]
                _log(verbose, 1,
                     f"[iter {iteration:4d}]  parallel  batch={len(batch)}  "
                     f"neg_pixels={len(neg_pixels)}  "
                     f"windows={min(batch_sizes)}-{max(batch_sizes)}")

                futures = {}
                for neg_idx, (cz, cy, cx), sub_size in batch:
                    d = sub_size // 2
                    window_counts[sub_size] += 1

                    phi_init_sub_flat = get_phi_sub_flat(
                        phi_init, cz, cy, cx, slice_shape, d)
                    phi_sub_flat = get_phi_sub_flat(
                        phi, cz, cy, cx, slice_shape, d)

                    is_at_edge, window_reached_max = _edge_flags(
                        cy, cx, d, slice_shape, max_window)

                    fut = executor.submit(
                        _optimize_single_window,
                        phi_sub_flat, phi_init_sub_flat, sub_size,
                        is_at_edge, window_reached_max,
                        threshold, max_minimize_iter, methodName,
                    )
                    futures[fut] = (neg_idx, cz, cy, cx, sub_size)

                batch_time = 0.0
                for fut in as_completed(futures):
                    neg_idx, cz, cy, cx, sub_size = futures[fut]
                    result_x, elapsed = fut.result()
                    batch_time = max(batch_time, elapsed)
                    _apply_result(phi, result_x, cy, cx, sub_size)

                iter_times.append(batch_time)

                jacobian_matrix = jacobian_det2D(phi)
                cur_neg = int((jacobian_matrix <= 0).sum())
                cur_min = float(jacobian_matrix.min())
                num_neg_jac.append(cur_neg)
                min_jdet_list.append(cur_min)
                error_list.append(np.sqrt(np.sum((phi - phi_init) ** 2)))

                cur_err = error_list[-1]
                _log(verbose, 1,
                     f"         -> neg_jdet {cur_neg:5d}  "
                     f"min_jdet {cur_min:+.6f}  L2 {cur_err:.4f}")

            # Per-step snapshot
            if plot_every and iteration % plot_every == 0:
                from modules.dvfviz import plot_step_snapshot
                cur_neg = int((jacobian_matrix <= 0).sum())
                cur_min = float(jacobian_matrix.min())
                plot_step_snapshot(jacobian_matrix, iteration, cur_neg, cur_min)

            if float(jacobian_matrix.min()) > threshold - err_tol:
                _log(verbose, 1,
                     f"[done] All Jdet > threshold after iter {iteration}")
                break

    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    end_time = time.time()
    elapsed = end_time - start_time

    final_err = np.sqrt(np.sum((phi - phi_init) ** 2))
    final_neg = int((jacobian_matrix <= 0).sum())
    final_min = float(jacobian_matrix.min())

    _log(verbose, 1, "")
    _log(verbose, 1, "=" * 60)
    _log(verbose, 1, f"  SUMMARY  ({methodName} — hybrid parallel)")
    _log(verbose, 1, "-" * 60)
    _log(verbose, 1, f"  Grid size        : {H} x {W}")
    _log(verbose, 1, f"  Iterations       : {iteration}  "
         f"(serial={serial_iters}, parallel={parallel_iters})")
    _log(verbose, 1, f"  Neg-Jdet  {init_neg:>5d} -> {final_neg:>5d}")
    _log(verbose, 1, f"  Min Jdet  {init_min:+.6f} -> {final_min:+.6f}")
    _log(verbose, 1, f"  L2 error         : {final_err:.6f}")
    _log(verbose, 1, f"  Time             : {elapsed:.2f}s")
    _log(verbose, 1, "=" * 60)

    num_neg_jac.append(final_neg)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        output_text = "Settings:\n"
        output_text += f"\tMethod: {methodName} (hybrid parallel)\n"
        output_text += f"\tThreshold: {threshold}\n"
        output_text += f"\tError tolerance: {err_tol}\n"
        output_text += f"\tMax iterations: {max_iterations}\n"
        output_text += f"\tMax per index iterations: {max_per_index_iter}\n"
        output_text += f"\tMax minimize iterations: {max_minimize_iter}\n"
        output_text += f"\tStarting window size: {starting_window_size + 2}\n\n"

        output_text += "Results:\n"
        output_text += f"\tInput deformation field resolution (height x width): {H} x {W}\n"
        output_text += f"\tTotal run-time: {elapsed} seconds\n"
        output_text += f"\tFinal L2 error: {final_err}\n"
        output_text += f"\tStarting number of non-positive Jacobian determinants: {init_neg}\n"
        output_text += f"\tFinal number of non-positive Jacobian determinants: {final_neg}\n"
        output_text += f"\tStarting Jacobian determinant minimum value: {init_min}\n"
        output_text += f"\tFinal Jacobian determinant minimum value: {final_min}\n"
        output_text += f"\tNumber of index iterations: {iteration}\n"
        output_text += f"\tSerial iterations: {serial_iters}\n"
        output_text += f"\tParallel iterations: {parallel_iters}"

        with open(save_path + "/results.txt", "w") as f:
            f.write(output_text)

        np.save(save_path + "/phi.npy", phi)
        np.save(save_path + "/error_list.npy", error_list)
        np.save(save_path + "/num_neg_jac.npy", num_neg_jac)
        np.save(save_path + "/iter_times.npy", iter_times)
        np.save(save_path + "/min_jdet_list.npy", min_jdet_list)

        with open(save_path + "/window_counts.csv", "w") as f:
            f.write("window_size,count\n")
            for ws in sorted(window_counts):
                f.write(f"{ws},{window_counts[ws]}\n")

    return phi
