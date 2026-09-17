"""
HealthConnect ML Pipeline — Model Training
Week 5 — Machine Learning Engineering Track

Combines preprocessing + Logistic Regression baseline into a single
scikit-learn Pipeline (Handoff Section 9/12/14), trains it on a stratified
80/20 split, and saves the fitted pipeline + metadata to disk.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.data.load_data import load_appointment_data
from src.data.preprocess import clean_dataset
from src.data.validate_data import validate_dataset
from src.features.feature_engineering import engineer_features
from src.models.model_utils import build_preprocessor, select_model_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_dataframe(raw_path=None) -> pd.DataFrame:
    """Run ingestion -> validation -> cleaning -> feature engineering."""
    raw_path = raw_path or config.RAW_DATA_PATH
    df = load_appointment_data(raw_path)
    report = validate_dataset(df)
    if not report.passed:
        raise RuntimeError(f"Dataset failed validation: {report.critical_issues}")

    df = clean_dataset(df)
    df = engineer_features(df)
    return df


def split_data(df: pd.DataFrame):
    """
    Stratified 80/20 split on the binary target, random_state=42
    (Handoff Section 7). Group-aware splitting on patient_id is flagged as
    a Week 6 upgrade (Handoff Section 18) — not implemented here since
    patient_id is dropped during cleaning, before this point.
    """
    X = select_model_features(df)
    y = df["target"]
    stratify = y if config.STRATIFY_ON_TARGET else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test


def build_model_pipeline() -> Pipeline:
    preprocessor = build_preprocessor()
    model = LogisticRegression(**config.MODEL_HYPERPARAMETERS)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_baseline_model(raw_path=None) -> tuple[Pipeline, dict, tuple]:
    """
    End-to-end: prepare data, split, train the Logistic Regression baseline,
    and return the fitted pipeline plus a metadata dict describing the run.
    """
    df = prepare_dataframe(raw_path)
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_model_pipeline()
    logger.info("Training Logistic Regression baseline on %d rows", len(X_train))
    pipeline.fit(X_train, y_train)

    metadata = {
        "model_type": config.MODEL_TYPE,
        "hyperparameters": config.MODEL_HYPERPARAMETERS,
        "features": config.ALL_MODEL_FEATURES,
        "target_mapping": config.TARGET_MAPPING,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": config.RANDOM_STATE,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "reminder_* and previous_*/prior_no_show_rate features are marked "
            "CONDITIONAL in configs/config.yaml pending confirmation of "
            "prediction-timing assumptions per the Week 5 Data Science handoff, "
            "Section 3/20."
        ),
    }
    return pipeline, metadata, (X_test, y_test)


def save_model(pipeline: Pipeline, metadata: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    with open(config.MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved model to %s", config.MODEL_PATH)
    logger.info("Saved metadata to %s", config.MODEL_METADATA_PATH)


if __name__ == "__main__":
    fitted_pipeline, run_metadata, test_data = train_baseline_model()
    save_model(fitted_pipeline, run_metadata)
    print(json.dumps(run_metadata, indent=2, default=str))
