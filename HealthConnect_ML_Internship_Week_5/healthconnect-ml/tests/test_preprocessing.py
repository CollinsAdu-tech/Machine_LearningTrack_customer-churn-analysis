"""
Unit tests for src/data/preprocess.py

Run with:
    pytest tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.load_data import load_appointment_data
from src.data.preprocess import clean_dataset

REAL_DATA_PATH = config.RAW_DATA_PATH


@pytest.fixture(scope="module")
def raw_df():
    return load_appointment_data(REAL_DATA_PATH)


@pytest.fixture(scope="module")
def cleaned_df(raw_df):
    return clean_dataset(raw_df)


def test_cancelled_rows_dropped(cleaned_df):
    # 5000 total, 263 Cancelled -> 4737 remain (verified against DS handoff)
    assert len(cleaned_df) == 4737
    assert "Cancelled" not in cleaned_df[config.TARGET_COLUMN].unique()


def test_target_encoded_correctly(cleaned_df):
    assert set(cleaned_df["target"].unique()) == {0, 1}
    no_show_rows = cleaned_df[cleaned_df[config.TARGET_COLUMN] == "No-Show"]
    assert (no_show_rows["target"] == 1).all()
    attended_rows = cleaned_df[cleaned_df[config.TARGET_COLUMN] == "Attended"]
    assert (attended_rows["target"] == 0).all()


def test_leakage_and_identifier_columns_dropped(cleaned_df):
    for col in config.IDENTIFIER_COLUMNS + config.LEAKAGE_COLUMNS + config.REDUNDANT_COLUMNS:
        assert col not in cleaned_df.columns


def test_reminder_channel_no_nulls_after_cleaning(cleaned_df):
    assert cleaned_df["reminder_channel"].isna().sum() == 0
    none_rows = cleaned_df[cleaned_df["reminder_channel"] == "None"]
    assert (none_rows["reminder_sent"] == "No").all()


def test_dates_parsed_to_datetime(cleaned_df):
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["booking_date"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["appointment_date"])
