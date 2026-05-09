import numpy as np
import pytest

from benchmarks.two_triangle import registry
import benchmarks.two_triangle.cases  # registers cases


SYNTHETIC_2D = (
    "synth2d_single_cell_flip",
    "synth2d_horizontal_bowtie",
    "synth2d_diagonal_bowtie",
    "synth2d_layered_bowtie_stack",
)


@pytest.mark.parametrize("name", SYNTHETIC_2D)
def test_synthetic_2d_case_registered(name):
    assert name in registry.list_cases()
    fn = registry.get_case(name)
    dvf, meta = fn()
    assert dvf.ndim == 4
    assert dvf.shape[0] == 3
    assert dvf.shape[1] == 1
    assert "title" in meta
    assert meta.get("dim") in (2, None)


@pytest.mark.parametrize("name", SYNTHETIC_2D)
def test_synthetic_2d_case_has_initial_folds(name):
    """Every synthetic case must start with at least one fold to be useful."""
    from benchmarks.two_triangle.metrics import fold_counts
    fn = registry.get_case(name)
    dvf, _ = fn()
    fc = fold_counts(dvf, threshold=0.01)
    assert fc["fold_count_jdet"] > 0 or fc["fold_count_tri"] > 0, (
        f"Case {name} has no initial folds — useless as a benchmark"
    )
