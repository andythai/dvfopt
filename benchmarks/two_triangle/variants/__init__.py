"""Importing this package registers all variants."""
from benchmarks.two_triangle.variants import baseline_serial         # noqa: F401
from benchmarks.two_triangle.variants import combinatorial_prefix    # noqa: F401
from benchmarks.two_triangle.variants import soft_margin             # noqa: F401
from benchmarks.two_triangle.variants import active_set              # noqa: F401
from benchmarks.two_triangle.variants import multigrid               # noqa: F401
from benchmarks.two_triangle.variants import trust_constr            # noqa: F401
