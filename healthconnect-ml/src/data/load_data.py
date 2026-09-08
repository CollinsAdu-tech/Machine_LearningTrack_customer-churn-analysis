"""
HealthConnect ML Pipeline — Data Loading
Week 5 — Machine Learning Engineering Track

Stage 1 (Data Ingestion) of the ML workflow (Week 4 ML System Design Doc).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when the dataset cannot be loaded or is empty."""


def load_appointment_data(path: str | Path) -> pd.DataFrame:
    """
    Load the HealthConnect appointment dataset from CSV.

    Raises
    ------
    FileNotFoundError
        If the given path does not exist.
    DataLoadError
        If the file cannot be parsed as CSV or is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surface a clear pipeline error
        raise DataLoadError(f"Failed to parse CSV at {path}: {exc}") from exc

    if df.empty:
        raise DataLoadError(f"Dataset at {path} loaded but contains 0 rows.")

    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], path)
    return df
