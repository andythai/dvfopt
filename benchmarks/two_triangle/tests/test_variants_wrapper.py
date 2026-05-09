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


@pytest.mark.parametrize("name", ["combinatorial_prefix"])
def test_wrapper_variant_runs(name):
    fn = registry.get_variant(name)
    dvf = _tiny_2d_fold()
    result = fn(dvf, threshold=0.01, max_iterations=20)
    assert result.phi_final.shape == (2, 6, 6)
    assert len(result.trajectory) >= 1


def test_combinatorial_prefix_reduces_or_preserves_folds():
    """Prefix step alone should not increase folds beyond initial."""
    from benchmarks.two_triangle.metrics import fold_counts
    fn = registry.get_variant("combinatorial_prefix")
    dvf = _tiny_2d_fold()
    init_fc = fold_counts(dvf, threshold=0.01)
    result = fn(dvf, threshold=0.01, max_iterations=0)  # only prefix, no SLSQP
    final_fc = fold_counts(result.phi_final, threshold=0.01)
    assert final_fc["fold_count_tri"] <= init_fc["fold_count_tri"]
