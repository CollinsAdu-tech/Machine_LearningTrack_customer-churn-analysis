"""
Unit tests for src/data/load_data.py and src/data/validate_data.py

Run with:
    pytest tests/test_data.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.load_data import DataLoadError, load_appointment_data
from src.data.validate_data import (
    DataValidationError,
    EXPECTED_COLUMNS,
    validate_dataset,
)

REAL_DATA_PATH = config.RAW_DATA_PATH


def make_valid_row(**overrides) -> dict:
    row = {
        "appointment_id": "HC-00001",
        "patient_id": "P-0001",
        "gender": "Female",
        "age": 40,
        "age_group": "35-44",
        "appointment_type": "Follow-up",
        "booking_date": "2/6/2025",
        "appointment_date": "2/18/2025",
        "appointment_day": "Tuesday",
        "appointment_time": "Afternoon",
        "booking_lead_days": 12,
        "previous_appointments": 2,
        "previous_no_shows": 0,
        "reminder_sent": "Yes",
        "reminder_channel": "WhatsApp",
        "distance_to_clinic_km": 19.3,
        "waiting_time_minutes": 29,
        "appointment_outcome": "No-Show",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Loading tests
# ---------------------------------------------------------------------------

def test_load_appointment_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_appointment_data("does/not/exist.csv")


def test_load_appointment_data_real_file():
    df = load_appointment_data(REAL_DATA_PATH)
    assert len(df) == 5000
    assert list(df.columns) == EXPECTED_COLUMNS


def test_load_appointment_data_empty_csv(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("appointment_id,patient_id\n")
    with pytest.raises(DataLoadError):
        load_appointment_data(empty_file)


# ---------------------------------------------------------------------------
# Validation tests — happy path
# ---------------------------------------------------------------------------

def test_validate_clean_dataframe_passes():
    df = pd.DataFrame([make_valid_row(), make_valid_row(appointment_id="HC-00002")])
    report = validate_dataset(df)
    assert report.passed
    assert report.duplicate_appointment_ids == 0
    assert report.invalid_outcome_values == 0


def test_validate_real_dataset_passes():
    df = load_appointment_data(REAL_DATA_PATH)
    report = validate_dataset(df)
    assert report.passed
    assert report.missing_value_counts.get("reminder_channel", 0) > 0
    assert report.missing_value_counts.get("distance_to_clinic_km", 0) > 0
    assert report.missing_value_counts.get("waiting_time_minutes", 0) > 0


# ---------------------------------------------------------------------------
# Validation tests — failure / warning conditions
# ---------------------------------------------------------------------------

def test_validate_missing_column_fails():
    df = pd.DataFrame([make_valid_row()]).drop(columns=["appointment_outcome"])
    report = validate_dataset(df)
    assert not report.passed
    assert "appointment_outcome" in report.missing_columns


def test_validate_missing_column_raises_when_requested():
    df = pd.DataFrame([make_valid_row()]).drop(columns=["appointment_outcome"])
    with pytest.raises(DataValidationError):
        validate_dataset(df, raise_on_critical=True)


def test_validate_detects_duplicate_appointment_ids():
    df = pd.DataFrame([make_valid_row(), make_valid_row()])
    report = validate_dataset(df)
    assert report.duplicate_appointment_ids == 1


def test_validate_detects_invalid_outcome_value():
    df = pd.DataFrame([make_valid_row(appointment_outcome="Rescheduled")])
    report = validate_dataset(df)
    assert report.invalid_outcome_values == 1


def test_validate_detects_no_show_gt_appointments():
    df = pd.DataFrame([make_valid_row(previous_appointments=1, previous_no_shows=3)])
    report = validate_dataset(df)
    assert report.no_show_gt_appointments_count == 1


def test_validate_detects_negative_values():
    df = pd.DataFrame([make_valid_row(waiting_time_minutes=-5)])
    report = validate_dataset(df)
    assert report.negative_value_issues.get("waiting_time_minutes") == 1


def test_validate_detects_invalid_dates():
    df = pd.DataFrame([make_valid_row(booking_date="not-a-date")])
    report = validate_dataset(df)
    assert report.invalid_dates == 1
