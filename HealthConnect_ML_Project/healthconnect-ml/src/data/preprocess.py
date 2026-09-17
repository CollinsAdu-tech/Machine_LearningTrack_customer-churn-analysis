"""
HealthConnect ML Pipeline — Data Cleaning / Preprocessing
Week 5 — Machine Learning Engineering Track

Implements the "Data Cleaning" stage of the pipeline (Handoff Section 13),
built directly on the Data Science Week 5 handoff:
    - Section 2: target definition (exclude Cancelled, map No-Show=1/Attended=0)
    - Section 3: leakage removal (waiting_time_minutes)
    - Section 6: reminder_channel structural-null handling, date parsing

This module produces a clean, correctly-typed dataframe with the target
filtered/encoded. Encoding/scaling for the model itself lives in
src/models/model_utils.py (the ColumnTransformer), not here.
"""

from __future__ import annotations

import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse booking_date and appointment_date to datetime64.

    Both columns are stored as text in %m/%d/%Y format with 0 verified
    parse failures on the full dataset (Handoff Section 1/6). Any row that
    fails to parse here is a genuine data-quality issue and is logged
    rather than silently dropped.
    """
    df = df.copy()
    for col in config.RAW_DATE_COLUMNS:
        parsed = pd.to_datetime(df[col], format="%m/%d/%Y", errors="coerce")
        n_failed = int(parsed.isna().sum() - df[col].isna().sum())
        if n_failed > 0:
            logger.warning("%d rows failed to parse in column %s", n_failed, col)
        df[col] = parsed
    return df


def filter_and_encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop Cancelled rows and encode the target as No-Show=1 / Attended=0.

    Per Handoff Section 2, this excludes ~5.3% of rows (Cancelled) and is
    flagged as a decision pending business/PM confirmation — applied here
    but documented, not silently baked in.
    """
    df = df.copy()
    before = len(df)
    df = df[df[config.TARGET_COLUMN].isin(config.TARGET_CLASSES_KEPT)].copy()
    after = len(df)
    logger.info(
        "Dropped %d rows with outcome '%s' (%d -> %d rows)",
        before - after,
        config.EXCLUDED_OUTCOME_VALUE,
        before,
        after,
    )
    df["target"] = df[config.TARGET_COLUMN].map(config.TARGET_MAPPING)
    return df


def handle_structural_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode reminder_channel's null values as an explicit "None" category.

    Per Handoff Section 6/16 (Finding 2), every null in reminder_channel
    corresponds exactly to reminder_sent == "No" — verified structural
    missingness, not a gap to impute.
    """
    df = df.copy()
    for col in config.EXPLICIT_NONE_CATEGORY_COLUMNS:
        df[col] = df[col].fillna(config.EXPLICIT_NONE_LABEL)
    return df


def drop_leakage_and_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that must never reach the model: identifiers, the
    confirmed-leakage waiting_time_minutes column, and age_group
    (redundant with age). Raw date columns are dropped later, after
    feature engineering has used them as a source.
    """
    df = df.copy()
    cols_to_drop = [
        c
        for c in (config.IDENTIFIER_COLUMNS + config.LEAKAGE_COLUMNS + config.REDUNDANT_COLUMNS)
        if c in df.columns
    ]
    return df.drop(columns=cols_to_drop)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning stage: validate happens before this (see
    validate_data.py), then clean -> feature engineer -> split ->
    preprocess (encode/scale) -> model.

    Note: raw date columns are intentionally KEPT after this step, because
    feature_engineering.py still needs appointment_date/booking_date as a
    source. They are dropped just before the model sees the data, inside
    model_utils.py's feature selection step.
    """
    df = parse_dates(df)
    df = filter_and_encode_target(df)
    df = handle_structural_nulls(df)
    df = drop_leakage_and_identifier_columns(df)
    return df
