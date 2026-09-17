"""
Unit tests for src/features/feature_engineering.py

Run with:
    pytest tests/test_features.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.load_data import load_appointment_data
from src.data.preprocess import clean_dataset
from src.features import feature_engineering

REAL_DATA_PATH = config.RAW_DATA_PATH


@pytest.fixture(scope="module")
def engineered_df():
    df = load_appointment_data(REAL_DATA_PATH)
    df = clean_dataset(df)
    return feature_engineering.engineer_features(df)


def test_all_engineered_columns_present(engineered_df):
    expected = [
        "prior_no_show_rate",
        "is_first_time_patient",
        "is_weekend_appointment",
        "lead_time_bucket",
        "distance_bucket",
        "reminder_sent_flag",
    ]
    for col in expected:
        assert col in engineered_df.columns


def test_prior_no_show_rate_no_nan_or_inf_for_first_time_patients(engineered_df):
    first_timers = engineered_df[engineered_df["previous_appointments"] == 0]
    assert not first_timers.empty
    assert not first_timers["prior_no_show_rate"].isna().any()
    assert np.isfinite(first_timers["prior_no_show_rate"]).all()
    assert (first_timers["prior_no_show_rate"] == 0.0).all()


def test_is_weekend_appointment_matches_day():
    df = pd.DataFrame({"appointment_day": ["Saturday", "Sunday", "Monday", "Friday"]})
    result = feature_engineering.add_is_weekend_appointment(df)
    assert result["is_weekend_appointment"].tolist() == [1, 1, 0, 0]


def test_reminder_sent_flag_matches_reminder_sent():
    df = pd.DataFrame({"reminder_sent": ["Yes", "No", "Yes"]})
    result = feature_engineering.add_reminder_sent_flag(df)
    assert result["reminder_sent_flag"].tolist() == [1, 0, 1]


def test_lead_time_bucket_assigns_expected_labels():
    df = pd.DataFrame({"booking_lead_days": [0, 10, 20, 45, 90]})
    result = feature_engineering.add_lead_time_bucket(df)
    assert result["lead_time_bucket"].tolist() == ["0-7", "8-14", "15-30", "31-60", "60+"]
