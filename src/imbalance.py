"""Class imbalance handling scenarios."""

from collections import Counter
from collections.abc import Iterable


def class_counts(labels: Iterable[int]) -> dict[int, int]:
    """Return per-class counts as a regular dictionary."""
    return dict(Counter(labels))
