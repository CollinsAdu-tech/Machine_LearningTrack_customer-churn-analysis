"""
HealthConnect ML Pipeline — End-to-End Orchestration
Week 5 — Machine Learning Engineering Track

Ties together every stage (ingestion -> validation -> cleaning -> feature
engineering -> split -> train -> evaluate -> save) behind a single
entry point, so the pipeline can be run as one command:

    python -m src.pipeline.pipeline

This does not introduce new logic — it sequences the modules in
src/data/, src/features/, and src/models/ in the order specified by the
Data Science handoff, Section 13 (Expected ML Pipeline).
"""

from __future__ import annotations

import json
import logging

from src import config
from src.data.load_data import load_appointment_data
from src.data.preprocess import clean_dataset
from src.data.validate_data import validate_dataset
from src.features.feature_engineering import engineer_features
from src.models.evaluate import evaluate_model
from src.models.train import build_model_pipeline, save_model, split_data
from src.models.model_utils import select_model_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(raw_path=None, save: bool = True) -> dict:
    """
    Run the full HealthConnect ML pipeline end-to-end and return a summary
    dict (validation report, training metadata, evaluation metrics).
    """
    raw_path = raw_path or config.RAW_DATA_PATH

    logger.info("Stage 1/2: Ingestion + Validation")
    df = load_appointment_data(raw_path)
    validation_report = validate_dataset(df)
    if not validation_report.passed:
        raise RuntimeError(f"Dataset failed validation: {validation_report.critical_issues}")

    logger.info("Stage 3: Data Cleaning")
    df = clean_dataset(df)

    logger.info("Stage 4: Feature Engineering")
    df = engineer_features(df)

    logger.info("Stage 5: Train/Test Split")
    X_train, X_test, y_train, y_test = split_data(df)

    logger.info("Stage 6: Model Training")
    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    logger.info("Stage 7: Model Evaluation")
    eval_result = evaluate_model(pipeline, X_test, y_test)
    logger.info("\n%s", eval_result.summary())

    metadata = {
        "model_type": config.MODEL_TYPE,
        "hyperparameters": config.MODEL_HYPERPARAMETERS,
        "features": config.ALL_MODEL_FEATURES,
        "target_mapping": config.TARGET_MAPPING,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": config.RANDOM_STATE,
    }

    if save:
        logger.info("Stage 8: Saving model + metadata")
        save_model(pipeline, metadata)

    return {
        "validation": validation_report.summary(),
        "training_metadata": metadata,
        "evaluation": eval_result.to_dict(),
    }


if __name__ == "__main__":
    summary = run_pipeline()
    print(json.dumps(summary["evaluation"], indent=2))
