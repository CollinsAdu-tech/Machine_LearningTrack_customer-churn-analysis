"""
HealthConnect ML Pipeline — Data Science Candidate Model Integration
Week 6 — Machine Learning Engineering Track

Integrates the two verified model artifacts received from Data Science in
the Week 6 handoff (HealthConnect_Week6_DataScience_Deliverable.md):
    - healthconnect_logreg_pipeline.joblib
    - healthconnect_gb_pipeline.joblib

Both are self-contained sklearn.Pipeline objects (ColumnTransformer +
classifier). DS's message described them as accepting a raw DataFrame with
"no separate encoding/scaling step needed." That claim was independently
verified and holds for encoding/scaling — but ONE additional upstream step
was found to be required and is NOT optional:

    CONFIRMED BUG: a raw NaN in `reminder_channel` (as read from CSV) is
    silently treated by the fitted OneHotEncoder as an unseen category
    (all-zero one-hot row) rather than matched to the fitted "None"
    category — even though "None" exists specifically to represent "no
    reminder was sent." This produces a materially different prediction
    (verified: 0.776 vs 0.789 probability on the same underlying record)
    with no error or warning. Confirmed present in BOTH pipelines, since
    they share the same ColumnTransformer architecture.

    `distance_to_clinic_km`'s raw NaN values, by contrast, ARE handled
    correctly by the bundled SimpleImputer — verified directly, no fix
    needed there.

This module makes the required fix structural: `prepare_candidate_input()`
is the single mandatory gate every record must pass through before either
candidate pipeline is called. Nothing about this module retrains or
modifies the Week 5 baseline (src/models/train.py, model_utils.py) — those
remain untouched, per the "preserve modularity, don't replace working
components without justified reason" rule.
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from src import config
from src.models.model_utils import risk_category_for_probability

logger = logging.getLogger(__name__)

# Raw fields a caller must supply to derive the candidate model's 11-feature
# schema. historical_no_show_rate is NOT in this list — it's derived, not
# supplied directly (see prepare_candidate_input).
REQUIRED_RAW_FIELDS = [
    "age",
    "booking_lead_days",
    "previous_appointments",
    "previous_no_shows",
    "distance_to_clinic_km",
    "gender",
    "appointment_type",
    "appointment_day",
    "appointment_time",
    "reminder_channel",
]


class CandidateModelInputError(Exception):
    """Raised when a request for a candidate model is malformed, missing
    fields, or contains a forbidden column."""


def _validate_request(record: dict) -> None:
    forbidden_present = [c for c in config.CANDIDATE_FORBIDDEN_COLUMNS if c in record]
    if forbidden_present:
        raise CandidateModelInputError(
            f"Request contains forbidden field(s) that must never be supplied at "
            f"prediction time: {forbidden_present}. These are either the training "
            f"target or a confirmed data-leakage column."
        )

    missing = [f for f in REQUIRED_RAW_FIELDS if f not in record]
    if missing:
        raise CandidateModelInputError(f"Request is missing required field(s): {missing}")


def compute_historical_no_show_rate(df: pd.DataFrame) -> pd.Series:
    """
    historical_no_show_rate = previous_no_shows / previous_appointments,
    with an explicit 0.0 fallback when previous_appointments == 0.

    This is the same computation as
    src.features.feature_engineering.add_prior_no_show_rate — reimplemented
    here under the name Data Science's candidate models expect
    (historical_no_show_rate), rather than importing and renaming, so this
    module has no hidden coupling to the Week 5 feature-engineering column
    names if those ever diverge further.
    """
    numerator = df[config.HISTORICAL_RATE_NUMERATOR]
    denominator = df[config.HISTORICAL_RATE_DENOMINATOR].replace(0, np.nan)
    rate = numerator / denominator
    return rate.fillna(config.HISTORICAL_RATE_FALLBACK)


def prepare_candidate_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    The single mandatory preparation gate for both candidate pipelines.

    Performs exactly two transformations, both required and verified
    against the actual artifacts (not assumed from the handoff text):
      1. Derive `historical_no_show_rate` from previous_no_shows /
         previous_appointments.
      2. Replace NaN in `reminder_channel` with the literal string "None"
         — REQUIRED, not optional. See module docstring for the confirmed
         bug this prevents.

    Does NOT one-hot encode, scale, or impute anything else — both
    candidate pipelines' bundled ColumnTransformer handles that correctly
    on its own (verified: distance_to_clinic_km NaN is correctly
    median-imputed by the bundled SimpleImputer).

    Returns a DataFrame containing exactly config.CANDIDATE_ALL_FEATURES,
    in a form safe to pass directly to either pipeline's .predict() /
    .predict_proba().
    """
    df = df.copy()
    df["historical_no_show_rate"] = compute_historical_no_show_rate(df)

    null_count = df["reminder_channel"].isna().sum()
    if null_count:
        logger.info(
            "Replacing %d null reminder_channel value(s) with the literal "
            "string '%s' (required — see module docstring for the "
            "confirmed OneHotEncoder bug this prevents).",
            null_count,
            config.CANDIDATE_REMINDER_CHANNEL_NULL_REPLACEMENT,
        )
    df["reminder_channel"] = df["reminder_channel"].fillna(
        config.CANDIDATE_REMINDER_CHANNEL_NULL_REPLACEMENT
    )

    missing = [c for c in config.CANDIDATE_ALL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Expected candidate-model feature columns missing: {missing}")

    return df[config.CANDIDATE_ALL_FEATURES].copy()


def load_candidate_model(model_name: str):
    """
    Load a candidate pipeline by name: "logistic_regression" or
    "gradient_boosting".
    """
    paths = {
        "logistic_regression": config.CANDIDATE_LOGREG_PATH,
        "gradient_boosting": config.CANDIDATE_GB_PATH,
    }
    if model_name not in paths:
        raise ValueError(
            f"Unknown candidate model '{model_name}'. Expected one of {list(paths)}."
        )
    path = paths[model_name]
    if not path.exists():
        raise FileNotFoundError(f"Candidate model artifact not found at {path}.")
    return joblib.load(path)


def _validate_output(proba: float) -> None:
    """
    Output validation per the engineering rules: probability must be in
    [0, 1] and must not be NaN.
    """
    if proba is None or (isinstance(proba, float) and np.isnan(proba)):
        raise ValueError("Model produced a NaN probability — refusing to return an invalid result.")
    if not (0.0 <= proba <= 1.0):
        raise ValueError(f"Model produced an out-of-range probability: {proba}")


def predict_with_candidate(record: dict, model_name: str, pipeline=None) -> dict:
    """
    Predict the no-show outcome for a single appointment record using one
    of the Data Science candidate pipelines.

    Parameters
    ----------
    record : dict
        Raw fields matching REQUIRED_RAW_FIELDS. Must NOT contain
        'appointment_outcome' or 'waiting_time_minutes'.
    model_name : str
        "logistic_regression" or "gradient_boosting".
    pipeline : fitted sklearn Pipeline, optional
        If not provided, loads the artifact named by model_name.

    Returns
    -------
    dict with keys: prediction (0/1), label ("Attended"/"No-Show"),
    no_show_probability (float in [0, 1]), risk_category, model_used.
    """
    _validate_request(record)

    if pipeline is None:
        pipeline = load_candidate_model(model_name)

    df = pd.DataFrame([record])
    X = prepare_candidate_input(df)

    proba = float(pipeline.predict_proba(X)[0, 1])
    _validate_output(proba)
    label_int = int(proba >= 0.5)

    return {
        "prediction": label_int,
        "label": "No-Show" if label_int == 1 else "Attended",
        "no_show_probability": round(proba, 4),
        "risk_category": risk_category_for_probability(proba),
        "model_used": model_name,
    }
