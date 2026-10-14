"""Small statistics helpers shared by the eval metrics and the cost report."""

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
