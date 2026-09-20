"""
HealthConnect ML Pipeline — Model Utilities
Week 5 — Machine Learning Engineering Track

Shared building blocks for train.py, evaluate.py, and predict.py:
    - select_model_features(): the boundary between "cleaned + engineered
      dataframe" and "what the model actually sees"
    - build_preprocessor(): the scikit-learn ColumnTransformer specified in
      the Data Science handoff, Section 6/12

Using a fitted ColumnTransformer (rather than manual pandas transforms)
makes the "fit on train fold only" requirement structural, not a
convention that can be silently violated (Handoff Section 7/15).
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the columns in the final feature set (Handoff Section 5/12).
    Raw date columns, identifiers, and the original target/outcome text
    column are excluded here.
    """
    missing = [c for c in config.ALL_MODEL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Expected feature columns missing before model: {missing}")
    return df[config.ALL_MODEL_FEATURES].copy()


def build_preprocessor() -> ColumnTransformer:
    """
    Build the (unfitted) preprocessing ColumnTransformer.

    Numerical pipeline: median-impute (only distance_to_clinic_km actually
    has nulls; SimpleImputer is a no-op for the rest) -> StandardScaler.
    Categorical pipeline: one-hot encode, ignoring unseen categories at
    inference time rather than raising.
    Boolean features pass through unchanged.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, config.NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
            ("bool", "passthrough", config.BOOLEAN_FEATURES),
        ]
    )
    return preprocessor


def risk_category_for_probability(proba: float) -> str:
    """Map a No-Show probability to Low/Medium/High per the Week 4 design doc."""
    if proba <= config.RISK_LOW_MAX:
        return "Low"
    elif proba <= config.RISK_MEDIUM_MAX:
        return "Medium"
    return "High"


def find_null_fields(record: dict, required_fields: list[str]) -> list[str]:
    """
    Week 7 addition. Return required fields whose value is None.

    Discovered during Week 7 edge-case testing: a field being *present* in
    the request dict (checked elsewhere) does not mean its value isn't
    None. Before this check existed, {"age": None, ...} passed the
    "missing field" check (the key exists) and was silently fed straight
    into the model, producing a prediction with no error or warning. This
    closes that gap.
    """
    return [f for f in required_fields if f in record and record[f] is None]


def find_non_numeric_fields(record: dict, numeric_fields: list[str]) -> list[str]:
    """
    Week 7 addition. Return numeric fields whose supplied value is not
    actually a number.

    Discovered during Week 7 edge-case testing: passing a string (e.g.
    age="thirty-nine") was not caught by our own validation — it
    propagated all the way into sklearn's SimpleImputer, which raised a
    raw, unfriendly ValueError ("Cannot use median strategy with
    non-numeric data") with no indication of which field or model caused
    it. This lets predict.py/candidate_models.py raise their own named
    error before the bad value ever reaches the model.

    bool is deliberately excluded from "non-numeric" (bool is a subclass
    of int in Python, and there's no legitimate boolean value among these
    fields anyway, so a stray True/False should still be flagged).
    """
    bad = []
    for f in numeric_fields:
        if f not in record or record[f] is None:
            continue  # handled by find_null_fields / missing-field check
        value = record[f]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            bad.append(f)
    return bad
