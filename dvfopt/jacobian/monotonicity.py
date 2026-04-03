"""Monotonicity (global injectivity) helpers for deformation fields."""

import numpy as np

from dvfopt._defaults import _unpack_size


def _monotonicity_diffs_2d(dy, dx):
    """Forward-difference monotonicity metrics for deformed coordinates.

    Returns ``(h_mono, v_mono)`` with shapes ``(H, W-1)`` and ``(H-1, W)``.
    """
    h_mono = 1.0 + np.diff(dx, axis=1)   # (H, W-1)
    v_mono = 1.0 + np.diff(dy, axis=0)   # (H-1, W)
    return h_mono, v_mono


def injectivity_constraint(phi_xy, submatrix_size, exclude_boundaries=True):
    """Return flattened monotonicity diffs for the SLSQP injectivity constraint."""
    sy, sx = _unpack_size(submatrix_size)
    pixels = sy * sx
    dx = phi_xy[:pixels].reshape((sy, sx))
    dy = phi_xy[pixels:].reshape((sy, sx))
    h_mono, v_mono = _monotonicity_diffs_2d(dy, dx)
    if exclude_boundaries:
        h_vals = h_mono[1:-1, 1:-1].flatten()
        v_vals = v_mono[1:-1, 1:-1].flatten()
    else:
        h_vals = h_mono.flatten()
        v_vals = v_mono.flatten()
    return np.concatenate([h_vals, v_vals])
