"""
HealthConnect ML Pipeline — Feature Engineering
Week 5 — Machine Learning Engineering Track

Implements the six engineered features specified in the Data Science Week 5
handoff, Section 4:
    prior_no_show_rate, is_first_time_patient, is_weekend_appointment,
    lead_time_bucket, distance_bucket, reminder_sent_flag

All six are built purely from existing columns, applied identically at
training and inference time (Handoff Section 13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def add_prior_no_show_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    previous_no_shows / previous_appointments, with an explicit fallback of
    0.0 when previous_appointments == 0 (avoids division by zero / NaN,
    matching the Handoff Section 15 test requirement).
    """
    df = df.copy()
    denom = df["previous_appointments"].replace(0, np.nan)
    rate = df["previous_no_shows"] / denom
    df["prior_no_show_rate"] = rate.fillna(0.0)
    return df


def add_is_first_time_patient(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_first_time_patient"] = (df["previous_appointments"] == 0).astype(int)
    return df


def add_is_weekend_appointment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_weekend_appointment"] = df["appointment_day"].isin(config.WEEKEND_DAYS).astype(int)
    return df


def add_lead_time_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lead_time_bucket"] = pd.cut(
        df["booking_lead_days"],
        bins=config.LEAD_TIME_BINS,
        labels=config.LEAD_TIME_LABELS,
    ).astype(str)
    return df


def add_distance_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_bucket"] = pd.cut(
        df["distance_to_clinic_km"],
        bins=config.DISTANCE_BINS,
        labels=config.DISTANCE_LABELS,
    ).astype(str)
    return df


def add_reminder_sent_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["reminder_sent_flag"] = (df["reminder_sent"] == "Yes").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all six engineered features in sequence.

    Expects a dataframe that has already been through
    src.data.preprocess.clean_dataset() (structural nulls handled, dates
    parsed, target encoded) but still retains the raw source columns
    (previous_appointments, booking_lead_days, distance_to_clinic_km,
    appointment_day, reminder_sent) needed to build these features.

    Known limitation (Week 6 candidate): distance_bucket here uses the
    pre-imputation distance value, so the 90 rows missing distance get a
    "nan" string bucket rather than one based on the imputed value. The
    one-hot encoder treats this as its own category, so the pipeline still
    runs correctly, but this is a refinement opportunity, not a bug fix
    required for the baseline.
    """
    df = add_prior_no_show_rate(df)
    df = add_is_first_time_patient(df)
    df = add_is_weekend_appointment(df)
    df = add_lead_time_bucket(df)
    df = add_distance_bucket(df)
    df = add_reminder_sent_flag(df)
    return df
