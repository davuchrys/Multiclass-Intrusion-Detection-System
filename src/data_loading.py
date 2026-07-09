"""Dataset loading and inspection utilities."""

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    return pd.read_csv(path, **kwargs)
