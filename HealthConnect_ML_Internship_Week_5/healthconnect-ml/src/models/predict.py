"""
HealthConnect ML Pipeline — Prediction Module
Week 5 baseline + Week 6 candidate model routing — Machine Learning
Engineering Track

Implements the input/output contract from the Data Science handoff
(Section 11, plus the rejection rules from Section 15) for the Week 5
baseline model, PLUS a routing layer added in Week 6 so callers can select
one of the Data Science candidate models (logistic_regression /
gradient_boosting, from src/models/candidate_models.py) through the same
predict() entry point.

Design decision: predict() is now a thin dispatcher. The Week 5 baseline
logic (16-feature model, engineer_features(), select_model_features()) is
UNCHANGED — moved into _predict_baseline() verbatim, not modified — so the
existing baseline behavior and its test suite (tests/test_model.py) keep
working exactly as before. model_name defaults to "baseline" specifically
so every pre-existing call to predict(record) or predict(record, pipeline)
is unaffected; this is a purely additive change.
    - Reject any request that contains appointment_outcome or
      waiting_time_minutes (the target and the confirmed leakage column
      must never be usable inputs, even by accident).
    - Reject a request missing a required raw field with a specific,
      named error rather than a generic pandas KeyError.
    - Return a probability in [0, 1] and a label in {0, 1}.
"""

from __future__ import annotations

import logging

import joblib
import pandas as pd

from src import config
from src.features.feature_engineering import engineer_features
from src.models.candidate_models import (
    CandidateModelInputError,
    predict_with_candidate,
)
from src.models.model_utils import risk_category_for_probability, select_model_features

logger = logging.getLogger(__name__)

CANDIDATE_MODEL_NAMES = ("logistic_regression", "gradient_boosting")

# Raw fields the caller must supply for the Week 5 BASELINE model — the
# pre-engineering source columns needed both directly by the model and to
# derive the six engineered features (Handoff Section 11 input contract).
# This list is specific to the baseline's 16-feature schema; the candidate
# models use a distinct, smaller schema defined in candidate_models.py.
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


def _predict_baseline(record: dict, pipeline=None) -> dict:
    """
    Week 5 baseline prediction. UNCHANGED from the original predict()
    implementation — only renamed and moved behind the dispatcher below.
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
        "model_used": "baseline",
    }


def predict(record: dict, pipeline=None, model_name: str = "baseline") -> dict:
    """
    Predict the no-show outcome for a single appointment record.

    Parameters
    ----------
    record : dict
        Raw appointment fields. Field requirements depend on model_name:
          - "baseline": REQUIRED_RAW_FIELDS in this module (11 fields,
            16-feature model after engineering).
          - "logistic_regression" / "gradient_boosting": the Week 6 Data
            Science candidate schema, see
            src.models.candidate_models.REQUIRED_RAW_FIELDS (10 fields,
            11-feature model). Passing baseline-only fields (e.g.
            reminder_sent) to a candidate model is harmless — they're
            simply ignored — but the candidate's own required fields must
            still be present.
        In all cases, must NOT contain 'appointment_outcome' or
        'waiting_time_minutes'.
    pipeline : fitted sklearn Pipeline, optional
        Only used when model_name="baseline". If not provided, loads the
        saved baseline model from config.MODEL_PATH. Ignored for candidate
        models — candidate_models.py manages its own artifact loading.
    model_name : str, default "baseline"
        One of "baseline", "logistic_regression", "gradient_boosting".
        Defaults to "baseline" so every pre-Week-6 call site is unaffected.

    Returns
    -------
    dict with keys: prediction (0/1), label ("Attended"/"No-Show"),
    no_show_probability (float in [0, 1]), risk_category (Low/Medium/High),
    model_used (str, which model actually produced this result).

    Raises
    ------
    PredictionInputError
        For baseline requests: forbidden or missing fields.
    CandidateModelInputError
        For candidate-model requests: forbidden or missing fields. Kept as
        a distinct exception type from PredictionInputError rather than
        silently unified, since the two models have different required
        fields and callers may need to distinguish which contract failed.
    ValueError
        If model_name is not one of the recognized options.
    """
    logger.info("Prediction requested — model_name=%s", model_name)

    try:
        if model_name == "baseline":
            result = _predict_baseline(record, pipeline=pipeline)
        elif model_name in CANDIDATE_MODEL_NAMES:
            result = predict_with_candidate(record, model_name=model_name)
        else:
            raise ValueError(
                f"Unknown model_name '{model_name}'. Expected 'baseline' or one of "
                f"{CANDIDATE_MODEL_NAMES}."
            )
    except (PredictionInputError, CandidateModelInputError, ValueError) as exc:
        # Log the failure type and model_name only — never the record
        # contents, which may include patient-relevant fields.
        logger.warning(
            "Prediction request rejected — model_name=%s, error_type=%s",
            model_name,
            type(exc).__name__,
        )
        raise

    logger.info(
        "Prediction completed — model_name=%s, label=%s, risk_category=%s",
        model_name,
        result["label"],
        result["risk_category"],
    )
    return result
