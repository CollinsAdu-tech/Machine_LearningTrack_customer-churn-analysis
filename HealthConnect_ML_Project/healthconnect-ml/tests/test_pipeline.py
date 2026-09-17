"""
Unit tests for src/pipeline/pipeline.py

Run with:
    pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.pipeline import run_pipeline


def test_run_pipeline_end_to_end_returns_expected_summary():
    summary = run_pipeline(save=False)

    assert "validation" in summary
    assert "training_metadata" in summary
    assert "evaluation" in summary

    meta = summary["training_metadata"]
    assert meta["train_rows"] + meta["test_rows"] == 4737

    ev = summary["evaluation"]
    assert 0.0 <= ev["recall"] <= 1.0
    assert 0.0 <= ev["precision"] <= 1.0
    assert 0.0 <= ev["roc_auc"] <= 1.0
    assert len(ev["confusion_matrix"]) == 2
    assert len(ev["confusion_matrix"][0]) == 2


def test_run_pipeline_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        run_pipeline(raw_path="does/not/exist.csv", save=False)
