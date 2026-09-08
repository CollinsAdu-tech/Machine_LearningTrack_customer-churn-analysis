"""
HealthConnect ML Pipeline — Prediction Module
Week 5 — Machine Learning Engineering Track

Implements the input/output contract from the Data Science handoff,
Section 11, plus the rejection rules from Section 15 (Model tests):
    - Reject any request that contains appointment_outcome or
      waiting_time_minutes (the target and the confirmed leakage column
      must never be usable inputs, even by accident).
    - Reject a request missing a required raw field with a specific,
      named error rather than a generic pandas KeyError.
    - Return a probability in [0, 1] and a label in {0, 1}.
"""

from __future__ import annotations

import joblib
import pandas as pd

from src import config
from src.features.feature_engineering import engineer_features
from src.models.model_utils import risk_category_for_probability, select_model_features

# Raw fields the caller must supply — the pre-engineering source columns
# needed both directly by the model and to derive the six engineered
# features (Handoff Section 11 input contract).
REQUIRED_RAW_FIELDS = [
    "gender",
    "age",
    "appointment_type",
    "appointment_day",
    "appointment_time",
    "booking_lead_days",
    "previous_appointments",
    "previous_no_shows",
    "reminder_sent",
    "reminder_channel",
    "distance_to_clinic_km",
]


class PredictionInputError(Exception):
    """Raised when a prediction request is malformed, missing fields, or
    contains a forbidden column."""


def _validate_request(record: dict) -> None:
    forbidden_present = [c for c in config.FORBIDDEN_PREDICTION_COLUMNS if c in record]
    if forbidden_present:
        raise PredictionInputError(
            f"Request contains forbidden field(s) that must never be supplied at "
            f"prediction time: {forbidden_present}. These are either the training "
            f"target or a confirmed data-leakage column (see Data Science handoff, "
            f"Section 3)."
        )

    missing = [f for f in REQUIRED_RAW_FIELDS if f not in record]
    if missing:
        raise PredictionInputError(f"Request is missing required field(s): {missing}")


def _build_feature_row(record: dict) -> pd.DataFrame:
    df = pd.DataFrame([record])
    df = engineer_features(df)
    return select_model_features(df)


def load_model(model_path=None):
    model_path = model_path or config.MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `python -m src.models.train` first."
        )
    return joblib.load(model_path)


def predict(record: dict, pipeline=None) -> dict:
    """
    Predict the no-show outcome for a single appointment record.

    Parameters
    ----------
    record : dict
        Raw appointment fields matching REQUIRED_RAW_FIELDS. Must NOT
        contain 'appointment_outcome' or 'waiting_time_minutes'.
    pipeline : fitted sklearn Pipeline, optional
        If not provided, loads the saved model from config.MODEL_PATH.

    Returns
    -------
    dict with keys: prediction (0/1), label ("Attended"/"No-Show"),
    no_show_probability (float in [0, 1]), risk_category (Low/Medium/High
    per the Week 4 design doc's thresholds).
    """
    _validate_request(record)

    if pipeline is None:
        pipeline = load_model()

    X = _build_feature_row(record)
    proba = float(pipeline.predict_proba(X)[0, 1])
    label_int = int(proba >= 0.5)

    return {
        "prediction": label_int,
        "label": "No-Show" if label_int == 1 else "Attended",
        "no_show_probability": round(proba, 4),
        "risk_category": risk_category_for_probability(proba),
    }
