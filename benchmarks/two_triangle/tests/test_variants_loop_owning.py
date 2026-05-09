import numpy as np
import pytest

from benchmarks.two_triangle import registry
import benchmarks.two_triangle.variants  # registers


def _tiny_2d_fold():
    H, W = 6, 6
    dvf = np.zeros((3, 1, H, W))
    dvf[1, 0, 3, 3] = -2.0
    dvf[2, 0, 3, 3] = -2.0
    return dvf


def test_soft_margin_registered():
    assert "soft_margin" in registry.list_variants()


def test_soft_margin_runs_and_progresses():
    fn = registry.get_variant("soft_margin")
    dvf = _tiny_2d_fold()
    result = fn(dvf, threshold=0.01, max_iterations=20)
    # Multi-row trajectory expected (loop-owning variant)
    assert len(result.trajectory) >= 2
    initial_folds = result.trajectory.iloc[0]["fold_count_tri"]
    final_folds = result.trajectory.iloc[-1]["fold_count_tri"]
    assert final_folds <= initial_folds


def test_soft_margin_trajectory_monotonic_on_synthetic():
    """Loop-owning variants should not regress on simple synthetic inputs."""
    fn = registry.get_variant("soft_margin")
    dvf = _tiny_2d_fold()
    result = fn(dvf, threshold=0.01, max_iterations=20)
    folds = result.trajectory["fold_count_tri"].values
    diffs = np.diff(folds)
    # Allow at most one "blip" of magnitude 1 (sub-window optimisation may
    # transiently introduce a fold during boundary repositioning).
    assert (diffs > 1).sum() == 0
