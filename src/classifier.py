"""LightGBM classifier helpers."""

from lightgbm import LGBMClassifier


def build_lightgbm(config: dict) -> LGBMClassifier:
    """Build a LightGBM multiclass classifier from a config dictionary."""
    return LGBMClassifier(**config)
