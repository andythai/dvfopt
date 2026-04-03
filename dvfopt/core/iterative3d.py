"""Iterative SLSQP algorithm for 3D deformation field correction."""

import time

import numpy as np

from dvfopt._defaults import _log, _resolve_params, _unpack_size_3d
from dvfopt.core.spatial3d import argmin_worst_voxel
from dvfopt.core.solver import _setup_accumulators, _print_summary, _save_results
from dvfopt.core.solver3d import (
    _init_phi_3d,
    _update_metrics_3d,
    _full_grid_step_3d,
    _serial_fix_voxel,
)


def iterative_3d(
    deformation,
    methodName="SLSQP",
    verbose=1,
    save_path=None,
    threshold=None,
    err_tol=None,
    max_iterations=None,
    max_per_index_iter=None,
    max_minimize_iter=None,
):
    """Iterative SLSQP correction of negative Jacobian determinants in 3D.

    Parameters
    ----------
    deformation : ndarray, shape ``(3, D, H, W)``
        Input deformation field with channels ``[dz, dy, dx]``.
    methodName : str
        Optimiser method passed to ``scipy.optimize.minimize``.
    verbose : int
        ``0`` = silent, ``1`` = per-iteration progress, ``2`` = debug.
    save_path : str or None
        Directory to save results.  ``None`` disables saving.
    threshold, err_tol, max_iterations, max_per_index_iter,
    max_minimize_iter :
        Override the corresponding default parameters.

    Returns
    -------
    phi : ndarray, shape ``(3, D, H, W)``
        Corrected displacement field ``[dz, dy, dx]``.
    """
    p = _resolve_params(threshold=threshold, err_tol=err_tol,
                        max_iterations=max_iterations,
                        max_per_index_iter=max_per_index_iter,
                        max_minimize_iter=max_minimize_iter)
    threshold = p["threshold"]
    err_tol = p["err_tol"]
    max_iterations = p["max_iterations"]
    max_per_index_iter = p["max_per_index_iter"]
    max_minimize_iter = p["max_minimize_iter"]

    if verbose is True:
        verbose = 1
    elif verbose is False:
        verbose = 0

    error_list, num_neg_jac, iter_times, min_jdet_list, window_counts = _setup_accumulators()

    start_time = time.time()
    phi, phi_init, D, H, W = _init_phi_3d(deformation)
    volume_shape = (D, H, W)
    max_window = (D, H, W)

    _log(verbose, 1,
         f"[init] Grid {D}x{H}x{W}  |  threshold={threshold}  "
         f"|  method={methodName}")

    jacobian_matrix, init_neg, init_min = _update_metrics_3d(
        phi, phi_init, num_neg_jac, min_jdet_list)

    _log(verbose, 1,
         f"[init] Neg-Jdet voxels: {init_neg}  |  min Jdet: {init_min:.6f}")

    iteration = 0
    while (iteration < max_iterations
           and (jacobian_matrix <= threshold - err_tol).any()):
        iteration += 1

        neg_index = argmin_worst_voxel(jacobian_matrix)

        jacobian_matrix, subvolume_size, per_index_iter, (cz, cy, cx) = \
            _serial_fix_voxel(
                neg_index, phi, phi_init, jacobian_matrix,
                volume_shape, window_counts,
                max_per_index_iter, max_minimize_iter,
                max_window, threshold, err_tol, methodName, verbose,
                error_list, num_neg_jac, min_jdet_list, iter_times,
            )

        # Full-grid fallback for non-cubic grids
        sz, sy, sx = _unpack_size_3d(subvolume_size)
        if (sz >= D and sy >= H and sx >= W
                and not (D == H == W)
                and (jacobian_matrix <= threshold - err_tol).any()):
            iter_start = time.time()
            _full_grid_step_3d(phi, phi_init, D, H, W, threshold,
                               max_minimize_iter, methodName, verbose)
            iter_times.append(time.time() - iter_start)

            jacobian_matrix, cur_neg, cur_min = _update_metrics_3d(
                phi, phi_init, num_neg_jac, min_jdet_list, error_list)

        cur_neg = int((jacobian_matrix <= 0).sum())
        cur_min = float(jacobian_matrix.min())
        cur_err = error_list[-1] if error_list else 0.0
        _log(verbose, 1,
             f"[iter {iteration:4d}]  fix ({neg_index[0]:3d},"
             f"{neg_index[1]:3d},{neg_index[2]:3d})  "
             f"win {sz}x{sy}x{sx}  neg_jdet {cur_neg:5d}  "
             f"min_jdet {cur_min:+.6f}  L2 {cur_err:.4f}  "
             f"sub-iters {per_index_iter}")

        if float(jacobian_matrix.min()) > threshold - err_tol:
            _log(verbose, 1,
                 f"[done] All Jdet > threshold after iter {iteration}")
            break

    end_time = time.time()
    elapsed = end_time - start_time

    final_err = np.sqrt(np.sum((phi - phi_init) ** 2))
    final_neg = int((jacobian_matrix <= 0).sum())
    final_min = float(jacobian_matrix.min())

    _print_summary(verbose, f"{methodName} — 3D", (D, H, W), iteration,
                   init_neg, final_neg, init_min, final_min,
                   final_err, elapsed)

    num_neg_jac.append(final_neg)

    if save_path is not None:
        _save_results(
            save_path, method=methodName, threshold=threshold,
            err_tol=err_tol, max_iterations=max_iterations,
            max_per_index_iter=max_per_index_iter,
            max_minimize_iter=max_minimize_iter,
            grid_shape=(D, H, W), elapsed=elapsed, final_err=final_err,
            init_neg=init_neg, final_neg=final_neg, init_min=init_min,
            final_min=final_min, iteration=iteration, phi=phi,
            error_list=error_list, num_neg_jac=num_neg_jac,
            iter_times=iter_times, min_jdet_list=min_jdet_list,
            window_counts=window_counts,
        )

    return phi
