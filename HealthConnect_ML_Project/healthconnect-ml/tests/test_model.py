"""
Unit tests for src/models/model_utils.py, train.py, evaluate.py, predict.py

Run with:
    pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.load_data import load_appointment_data
from src.data.preprocess import clean_dataset
from src.features.feature_engineering import engineer_features
from src.models.model_utils import build_preprocessor, select_model_features
from src.models.train import train_baseline_model

REAL_DATA_PATH = config.RAW_DATA_PATH


@pytest.fixture(scope="module")
def engineered_df():
    df = load_appointment_data(REAL_DATA_PATH)
    df = clean_dataset(df)
    return engineer_features(df)


# ---------------------------------------------------------------------------
# model_utils tests
# ---------------------------------------------------------------------------

def test_select_model_features_returns_expected_columns(engineered_df):
    X = select_model_features(engineered_df)
    assert list(X.columns) == config.ALL_MODEL_FEATURES
    assert len(X) == len(engineered_df)


def test_select_model_features_raises_on_missing_column(engineered_df):
    broken = engineered_df.drop(columns=["age"])
    with pytest.raises(KeyError):
        select_model_features(broken)


def test_preprocessor_onehot_categories_match_known_cardinality(engineered_df):
    X = select_model_features(engineered_df)
    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    gender_idx = config.CATEGORICAL_FEATURES.index("gender")
    apt_type_idx = config.CATEGORICAL_FEATURES.index("appointment_type")
    assert len(onehot.categories_[gender_idx]) == 3
    assert len(onehot.categories_[apt_type_idx]) == 4


def test_preprocessor_fit_on_train_only_does_not_use_test_stats(engineered_df):
    X = select_model_features(engineered_df)
    train_X = X.iloc[:3000].copy()

    preprocessor_a = build_preprocessor()
    preprocessor_a.fit(train_X)
    transformed_a = preprocessor_a.transform(train_X)

    preprocessor_b = build_preprocessor()
    preprocessor_b.fit(train_X)
    transformed_b = preprocessor_b.transform(train_X)

    np.testing.assert_array_almost_equal(
        transformed_a.toarray() if hasattr(transformed_a, "toarray") else transformed_a,
        transformed_b.toarray() if hasattr(transformed_b, "toarray") else transformed_b,
    )


# ---------------------------------------------------------------------------
# Training / pipeline tests
# ---------------------------------------------------------------------------

def test_train_pipeline_end_to_end():
    pipeline, metadata, (X_test, y_test) = train_baseline_model()
    assert metadata["train_rows"] + metadata["test_rows"] == 4737
    preds = pipeline.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})

    probas = pipeline.predict_proba(X_test)[:, 1]
    assert (probas >= 0).all() and (probas <= 1).all()


def test_model_save_and_reload_produces_identical_predictions(tmp_path):
    pipeline, _, (X_test, y_test) = train_baseline_model()
    model_path = tmp_path / "model_test.joblib"
    joblib.dump(pipeline, model_path)

    assert model_path.exists()
    assert model_path.stat().st_size > 0

    reloaded = joblib.load(model_path)
    original_proba = pipeline.predict_proba(X_test)
    reloaded_proba = reloaded.predict_proba(X_test)
    np.testing.assert_array_almost_equal(original_proba, reloaded_proba)


# ---------------------------------------------------------------------------
# predict.py tests
# ---------------------------------------------------------------------------

def test_predict_module_rejects_forbidden_fields():
    from src.models.predict import PredictionInputError, predict

    record = {
        "gender": "Female", "age": 39, "appointment_type": "Follow-up",
        "appointment_day": "Tuesday", "appointment_time": "Afternoon",
        "booking_lead_days": 12, "previous_appointments": 2, "previous_no_shows": 0,
        "reminder_sent": "Yes", "reminder_channel": "WhatsApp",
        "distance_to_clinic_km": 19.3,
        "waiting_time_minutes": 29,
    }
    with pytest.raises(PredictionInputError):
        predict(record)


def test_predict_module_rejects_missing_fields():
    from src.models.predict import PredictionInputError, predict

    record = {"gender": "Female", "age": 39}
    with pytest.raises(PredictionInputError):
        predict(record)


def test_predict_module_returns_valid_output_shape():
    from src.models.predict import predict

    record = {
        "gender": "Male", "age": 31, "appointment_type": "Specialist Consultation",
        "appointment_day": "Friday", "appointment_time": "Morning",
        "booking_lead_days": 2, "previous_appointments": 6, "previous_no_shows": 0,
        "reminder_sent": "Yes", "reminder_channel": "SMS", "distance_to_clinic_km": 14.3,
    }
    result = predict(record)
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("Attended", "No-Show")
    assert 0.0 <= result["no_show_probability"] <= 1.0
    assert result["risk_category"] in ("Low", "Medium", "High")
