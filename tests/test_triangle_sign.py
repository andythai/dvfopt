"""Tests for per-pixel two-triangle sign-only Jacobian check."""

import numpy as np
import pytest

from dvfopt.jacobian import (
    jacobian_det2D,
    triangle_sign_det2D,
    triangle_sign_count_negatives,
    triangulated_shoelace_det2D,
)
from dvfopt.jacobian.shoelace import _all_triangle_areas_2d
from dvfopt.jacobian.triangle_sign import _triangle_areas_2d


def _make_phi(dy, dx):
    """Build a (2, H, W) phi with channels [dy, dx]."""
    return np.stack([dy, dx])


def test_identity_all_positive():
    H, W = 8, 10
    phi = np.zeros((2, H, W))
    signs = triangle_sign_det2D(phi)
    assert signs.shape == (2, H - 1, W - 1)
    assert signs.dtype == np.int8
    assert (signs == 1).all()


def test_uniform_translation_all_positive():
    H, W = 6, 7
    dy = np.full((H, W), 0.3)
    dx = np.full((H, W), -0.4)
    signs = triangle_sign_det2D(_make_phi(dy, dx))
    assert (signs == 1).all()


def test_positive_scale_all_positive():
    H, W = 6, 6
    yy, xx = np.mgrid[:H, :W].astype(float)
    dy = 0.1 * yy
    dx = 0.1 * xx
    signs = triangle_sign_det2D(_make_phi(dy, dx))
    assert (signs == 1).all()


def test_y_reflection_all_negative():
    """Reflecting across the x-axis (dy = -2y, dx = 0) inverts orientation.

    Full two-axis reflection (dx = -2x, dy = -2y) is a 180° rotation and
    preserves orientation, so we only flip one axis here.
    """
    H, W = 5, 5
    yy, _ = np.mgrid[:H, :W].astype(float)
    dy = -2.0 * yy
    dx = np.zeros((H, W))
    signs = triangle_sign_det2D(_make_phi(dy, dx))
    assert (signs == -1).all()


def test_shoelace_artifact_case():
    """Central diff is all-positive but two cells are geometrically folded."""
    H = W = 7
    dx = np.zeros((H, W))
    dy = np.zeros((H, W))
    dx[3, 3] = +1.2
    dx[3, 4] = -1.2
    phi = _make_phi(dy, dx)

    jdet = jacobian_det2D(phi)
    assert (jdet > 0).all(), "central diff should miss this fold"

    n_neg_triangle = triangle_sign_count_negatives(phi)
    assert n_neg_triangle > 0, "triangle sign check should catch the fold"

    signs = triangle_sign_det2D(phi)
    bad = np.argwhere(signs <= 0)
    assert bad.size > 0


def test_cross_check_all_triangle_strict_implies_our_two_positive():
    """If the strict 4-triangle check passes at a cell, our 2 must too.

    Our T1/T2 are the TR-BL split of each cell; the 4-triangle check
    enforces both diagonal splits, so ``(all 4 > 0)`` implies ``(T1 > 0 and
    T2 > 0)`` at that cell.
    """
    rng = np.random.default_rng(0)
    H, W = 12, 14
    dy = rng.normal(scale=0.08, size=(H, W))
    dx = rng.normal(scale=0.08, size=(H, W))

    phi = _make_phi(dy, dx)
    strict_areas = _all_triangle_areas_2d(dy, dx)  # shape (4, H-1, W-1)
    strict_all_pos = (strict_areas > 0).all(axis=0)  # (H-1, W-1)

    signs = triangle_sign_det2D(phi)
    our_all_pos = (signs > 0).all(axis=0)  # (H-1, W-1)

    assert (our_all_pos | ~strict_all_pos).all(), (
        "strict_all_pos => our_all_pos failed somewhere"
    )


def test_asymmetry_vs_tl_br_split():
    """Our TR-BL split and the existing TL-BR split disagree in principle.

    Push TL of cell (2, 2) toward the interior so it crosses the TR-BL
    diagonal but stays on the same side of the TL-BR diagonal. Only the
    TR-BL triangulation should flag the fold at this cell.
    """
    H = W = 5
    dy = np.zeros((H, W))
    dx = np.zeros((H, W))
    dx[2, 2] = 0.8  # TL of cell (2, 2) → (2.8, 2.8)
    dy[2, 2] = 0.8
    phi = _make_phi(dy, dx)

    tl_br = triangulated_shoelace_det2D(phi)  # existing TL-BR split
    tr_bl_signs = triangle_sign_det2D(phi)    # our TR-BL split

    # Cell (2, 2) under TL-BR split stays convex (both triangles positive)
    assert (tl_br[:, 2, 2] > 0).all(), "TL-BR split should keep (2,2) positive"
    # ...but at least one TR-BL triangle flips
    assert (tr_bl_signs[:, 2, 2] <= 0).any(), "TR-BL split should catch the fold"


def test_internal_area_matches_sign_function():
    rng = np.random.default_rng(1)
    H, W = 5, 6
    dy = rng.normal(scale=0.1, size=(H, W))
    dx = rng.normal(scale=0.1, size=(H, W))

    T1, T2 = _triangle_areas_2d(dy, dx)
    phi = _make_phi(dy, dx)
    signs = triangle_sign_det2D(phi)
    assert np.array_equal(signs[0], np.sign(T1).astype(np.int8))
    assert np.array_equal(signs[1], np.sign(T2).astype(np.int8))
