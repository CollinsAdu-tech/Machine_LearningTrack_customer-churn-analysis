"""
HealthConnect ML Pipeline — Config Loader
Week 5 — Machine Learning Engineering Track

Loads configs/config.yaml once and exposes it as a module-level dict plus a
few convenience constants, so every other module reads configuration from
one place rather than hardcoding values (Handoff Section 12/14).
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

# ---------------------------------------------------------------------------
# Convenience accessors (paths resolved relative to project root)
# ---------------------------------------------------------------------------

RAW_DATA_PATH = PROJECT_ROOT / CONFIG["paths"]["raw_data"]
PROCESSED_DATA_DIR = PROJECT_ROOT / CONFIG["paths"]["processed_data_dir"]
MODELS_DIR = PROJECT_ROOT / CONFIG["paths"]["models_dir"]
MODEL_PATH = MODELS_DIR / CONFIG["paths"]["model_filename"]
MODEL_METADATA_PATH = MODELS_DIR / CONFIG["paths"]["model_metadata_filename"]

TARGET_COLUMN = CONFIG["target"]["column"]
TARGET_CLASSES_KEPT = CONFIG["target"]["classes_kept"]
EXCLUDED_OUTCOME_VALUE = CONFIG["target"]["excluded_outcome_value"]
TARGET_MAPPING = CONFIG["target"]["mapping"]

IDENTIFIER_COLUMNS = CONFIG["columns_dropped_before_model"]["identifiers"]
LEAKAGE_COLUMNS = CONFIG["columns_dropped_before_model"]["leakage"]
REDUNDANT_COLUMNS = CONFIG["columns_dropped_before_model"]["redundant"]
RAW_DATE_COLUMNS = CONFIG["columns_dropped_before_model"]["raw_dates"]

FORBIDDEN_PREDICTION_COLUMNS = CONFIG["forbidden_prediction_columns"]

CONDITIONAL_SOURCE_COLUMNS = CONFIG["conditional_features"]["source_columns"]
CONDITIONAL_ENGINEERED_COLUMNS = CONFIG["conditional_features"]["engineered_columns"]

NUMERICAL_FEATURES = CONFIG["features"]["numerical"]
CATEGORICAL_FEATURES = CONFIG["features"]["categorical"]
BOOLEAN_FEATURES = CONFIG["features"]["boolean"]
ALL_MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

MEDIAN_IMPUTE_COLUMNS = CONFIG["missing_value_strategy"]["median_impute_columns"]
EXPLICIT_NONE_CATEGORY_COLUMNS = CONFIG["missing_value_strategy"]["explicit_none_category_columns"]
EXPLICIT_NONE_LABEL = CONFIG["missing_value_strategy"]["explicit_none_label"]

LEAD_TIME_BINS = CONFIG["feature_engineering"]["lead_time_bins"]
LEAD_TIME_LABELS = CONFIG["feature_engineering"]["lead_time_labels"]
DISTANCE_BINS = CONFIG["feature_engineering"]["distance_bins"]
DISTANCE_LABELS = CONFIG["feature_engineering"]["distance_labels"]
WEEKEND_DAYS = set(CONFIG["feature_engineering"]["weekend_days"])

TEST_SIZE = CONFIG["train_test_split"]["test_size"]
RANDOM_STATE = CONFIG["train_test_split"]["random_state"]
STRATIFY_ON_TARGET = CONFIG["train_test_split"]["stratify_on_target"]

MODEL_TYPE = CONFIG["model"]["type"]
MODEL_HYPERPARAMETERS = CONFIG["model"]["hyperparameters"]

PRIMARY_METRIC = CONFIG["evaluation"]["primary_metric"]
SECONDARY_METRICS = CONFIG["evaluation"]["secondary_metrics"]

RISK_LOW_MAX = CONFIG["risk_categories"]["low_max"]
RISK_MEDIUM_MAX = CONFIG["risk_categories"]["medium_max"]
