"""
Unit and integration tests for src/models/candidate_models.py

Covers the Week 6 Data Science candidate model integration:
    - loading both verified artifacts
    - the mandatory input-preparation gate (historical_no_show_rate,
      reminder_channel null handling)
    - the pinned regression test for the confirmed reminder_channel /
      OneHotEncoder bug (Data Science Week 7 requirement #6)
    - input/output contract validation
    - reload-identity and independently-reproduced evaluation metrics

Run with:
    pytest tests/test_candidate_models.py -v
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.models.candidate_models import (
    CandidateModelInputError,
    compute_historical_no_show_rate,
    load_candidate_model,
    prepare_candidate_input,
    predict_with_candidate,
)

RAW_DATA_PATH = config.RAW_DATA_PATH


def make_valid_record(**overrides) -> dict:
    record = {
        "age": 39,
        "booking_lead_days": 12,
        "previous_appointments": 2,
        "previous_no_shows": 0,
        "distance_to_clinic_km": 19.3,
        "gender": "Female",
        "appointment_type": "Follow-up",
        "appointment_day": "Tuesday",
        "appointment_time": "Afternoon",
        "reminder_channel": "WhatsApp",
    }
    record.update(overrides)
    return record


@pytest.fixture(scope="module")
def logreg_pipeline():
    return load_candidate_model("logistic_regression")


@pytest.fixture(scope="module")
def gb_pipeline():
    return load_candidate_model("gradient_boosting")


@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv(RAW_DATA_PATH)


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def test_both_artifacts_load_without_error(logreg_pipeline, gb_pipeline):
    assert logreg_pipeline is not None
    assert gb_pipeline is not None


def test_load_unknown_model_name_raises():
    with pytest.raises(ValueError):
        load_candidate_model("random_forest")


def test_load_missing_artifact_raises(tmp_path, monkeypatch):
    fake_path = tmp_path / "does_not_exist.joblib"
    monkeypatch.setattr(config, "CANDIDATE_LOGREG_PATH", fake_path)
    with pytest.raises(FileNotFoundError):
        load_candidate_model("logistic_regression")


# ---------------------------------------------------------------------------
# Fitted category verification (pins the exact schema both models expect)
# ---------------------------------------------------------------------------

def test_fitted_categories_match_config(logreg_pipeline, gb_pipeline):
    for pipeline in (logreg_pipeline, gb_pipeline):
        prep = pipeline.named_steps["prep"]
        cat_encoder = prep.named_transformers_["cat"]
        for col, expected_cats in config.CANDIDATE_FITTED_CATEGORIES.items():
            idx = config.CANDIDATE_CATEGORICAL_FEATURES.index(col)
            assert list(cat_encoder.categories_[idx]) == expected_cats


# ---------------------------------------------------------------------------
# Input preparation gate
# ---------------------------------------------------------------------------

def test_compute_historical_no_show_rate_matches_formula():
    df = pd.DataFrame({"previous_no_shows": [2, 0, 5], "previous_appointments": [4, 0, 5]})
    rate = compute_historical_no_show_rate(df)
    assert rate.tolist() == [0.5, 0.0, 1.0]


def test_prepare_candidate_input_returns_expected_columns():
    df = pd.DataFrame([make_valid_record()])
    prepared = prepare_candidate_input(df)
    assert list(prepared.columns) == config.CANDIDATE_ALL_FEATURES


def test_prepare_candidate_input_fills_reminder_channel_null():
    df = pd.DataFrame([make_valid_record(reminder_channel=None)])
    prepared = prepare_candidate_input(df)
    assert prepared["reminder_channel"].iloc[0] == "None"
    assert prepared["reminder_channel"].isna().sum() == 0


def test_prepare_candidate_input_raises_on_missing_source_column():
    df = pd.DataFrame([make_valid_record()]).drop(columns=["age"])
    with pytest.raises(KeyError):
        prepare_candidate_input(df)


# ---------------------------------------------------------------------------
# PINNED REGRESSION TEST — the confirmed reminder_channel / OneHotEncoder bug
# (Data Science Week 7 requirement #6: "regression test on leakage/null-
# handling controls, carried over from Week 5, still unimplemented as an
# automated test" — this is that test.)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["logistic_regression", "gradient_boosting"])
def test_raw_nan_reminder_channel_differs_from_prepared_input(model_name, raw_df):
    """
    PINS A CONFIRMED BUG: a raw NaN in reminder_channel is silently treated
    by the fitted OneHotEncoder as an unseen category (all-zero one-hot
    row), NOT matched to the fitted "None" category, even though "None"
    exists specifically to represent "no reminder was sent." This produces
    a materially different prediction with no error or warning.

    This test does NOT assert the bug is fixed inside the model artifact
    (it isn't — that's DS's ColumnTransformer, not ours to change). It
    asserts that prepare_candidate_input() is the thing standing between
    a raw record and this bug — i.e. that skipping preparation produces a
    measurably different (and wrong) result than going through it.

    If this test ever starts failing because the two probabilities become
    equal, it means the underlying artifacts changed (e.g. re-fit encoder
    that now handles NaN) — investigate before assuming the fix is now
    unnecessary.
    """
    pipeline = load_candidate_model(model_name)

    row_with_null = raw_df[raw_df["reminder_channel"].isna()].iloc[[0]].copy()
    row_with_null["historical_no_show_rate"] = compute_historical_no_show_rate(row_with_null)

    # Path A: skip the mandatory gate, feed raw NaN straight to the pipeline
    raw_input = row_with_null[config.CANDIDATE_ALL_FEATURES].copy()
    proba_raw_nan = pipeline.predict_proba(raw_input)[0, 1]

    # Path B: go through the required preparation gate
    prepared_input = prepare_candidate_input(row_with_null)
    proba_prepared = pipeline.predict_proba(prepared_input)[0, 1]

    assert not np.isclose(proba_raw_nan, proba_prepared), (
        f"Expected raw-NaN and prepared-input probabilities to differ for "
        f"{model_name} (confirmed bug), but they matched. Verify the "
        f"artifact hasn't changed before removing this test."
    )


def test_distance_nan_is_handled_correctly_by_bundled_imputer(gb_pipeline, raw_df):
    """
    Contrast case: unlike reminder_channel, a raw NaN in
    distance_to_clinic_km IS handled correctly by the bundled
    SimpleImputer with no preparation-gate intervention needed. This test
    documents that this column is NOT part of the required fix.
    """
    row = raw_df[raw_df["distance_to_clinic_km"].isna()].iloc[[0]].copy()
    row["historical_no_show_rate"] = compute_historical_no_show_rate(row)
    row["reminder_channel"] = row["reminder_channel"].fillna("None")

    X = row[config.CANDIDATE_ALL_FEATURES]
    proba = gb_pipeline.predict_proba(X)[0, 1]
    assert 0.0 <= proba <= 1.0  # succeeds without error; no NaN propagation


# ---------------------------------------------------------------------------
# Request validation (input contract)
# ---------------------------------------------------------------------------

def test_predict_rejects_forbidden_fields():
    record = make_valid_record(waiting_time_minutes=25)
    with pytest.raises(CandidateModelInputError):
        predict_with_candidate(record, "gradient_boosting")


def test_predict_rejects_appointment_outcome_field():
    record = make_valid_record(appointment_outcome="No-Show")
    with pytest.raises(CandidateModelInputError):
        predict_with_candidate(record, "gradient_boosting")


def test_predict_rejects_missing_required_field():
    record = make_valid_record()
    del record["distance_to_clinic_km"]
    with pytest.raises(CandidateModelInputError):
        predict_with_candidate(record, "gradient_boosting")


# ---------------------------------------------------------------------------
# Output validation (output contract)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["logistic_regression", "gradient_boosting"])
def test_predict_output_shape_and_ranges(model_name):
    record = make_valid_record()
    result = predict_with_candidate(record, model_name)
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("Attended", "No-Show")
    assert 0.0 <= result["no_show_probability"] <= 1.0
    assert result["risk_category"] in ("Low", "Medium", "High")
    assert result["model_used"] == model_name


def test_predict_handles_unseen_category_gracefully():
    record = make_valid_record(appointment_type="Telehealth Video Call")
    result = predict_with_candidate(record, "gradient_boosting")
    assert 0.0 <= result["no_show_probability"] <= 1.0


# ---------------------------------------------------------------------------
# Reload identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_name,fixture_name",
    [("logistic_regression", "logreg_pipeline"), ("gradient_boosting", "gb_pipeline")],
)
def test_reload_produces_identical_predictions(model_name, fixture_name, request, tmp_path):
    pipeline = request.getfixturevalue(fixture_name)
    reload_path = tmp_path / f"{model_name}_reload_test.joblib"
    joblib.dump(pipeline, reload_path)
    reloaded = joblib.load(reload_path)

    record = make_valid_record()
    df = pd.DataFrame([record])
    df["historical_no_show_rate"] = compute_historical_no_show_rate(df)
    X = prepare_candidate_input(df)

    original_proba = pipeline.predict_proba(X)
    reloaded_proba = reloaded.predict_proba(X)
    np.testing.assert_array_almost_equal(original_proba, reloaded_proba)


# ---------------------------------------------------------------------------
# Independently-reproduced evaluation metrics
# (confirms the DS handoff's reported numbers, doesn't just trust them)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["logistic_regression", "gradient_boosting"])
def test_reported_evaluation_metrics_are_reproducible(model_name, raw_df):
    pipeline = load_candidate_model(model_name)

    df = raw_df[raw_df["appointment_outcome"].isin(["Attended", "No-Show"])].copy()
    df["target"] = (df["appointment_outcome"] == "No-Show").astype(int)
    df = prepare_candidate_input(df.assign(
        historical_no_show_rate=compute_historical_no_show_rate(df)
    )).join(df[["patient_id", "target"]])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))
    test = df.iloc[test_idx]

    # 0 patient overlap is a correctness precondition for the comparison itself
    train = df.iloc[train_idx]
    assert len(set(train["patient_id"]) & set(test["patient_id"])) == 0

    X_test = test[config.CANDIDATE_ALL_FEATURES]
    y_test = test["target"]
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    expected = config.CANDIDATE_REPORTED_EVALUATION[model_name]
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    assert round(accuracy_score(y_test, preds), 4) == pytest.approx(expected["accuracy"], abs=0.0001)
    assert round(roc_auc_score(y_test, proba), 4) == pytest.approx(expected["roc_auc"], abs=0.0001)
    assert (tn, fp, fn, tp) == (
        expected["confusion_matrix"]["tn"],
        expected["confusion_matrix"]["fp"],
        expected["confusion_matrix"]["fn"],
        expected["confusion_matrix"]["tp"],
    )
