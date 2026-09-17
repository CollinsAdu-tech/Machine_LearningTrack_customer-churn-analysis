"""
Tests for the Week 6 dispatcher added to src/models/predict.py.

These tests are separate from tests/test_model.py (which covers the Week 5
baseline path, unchanged) and tests/test_candidate_models.py (which covers
candidate_models.py directly). This file specifically tests that
predict()'s routing logic sends each model_name to the right underlying
implementation and that results match calling that implementation directly
— i.e. the dispatcher adds no side effects of its own.

Run with:
    pytest tests/test_predict_dispatcher.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.candidate_models import CandidateModelInputError, predict_with_candidate
from src.models.predict import PredictionInputError, predict


def make_baseline_record(**overrides) -> dict:
    record = {
        "gender": "Female", "age": 39, "appointment_type": "Follow-up",
        "appointment_day": "Tuesday", "appointment_time": "Afternoon",
        "booking_lead_days": 12, "previous_appointments": 2, "previous_no_shows": 0,
        "reminder_sent": "Yes", "reminder_channel": "WhatsApp", "distance_to_clinic_km": 19.3,
    }
    record.update(overrides)
    return record


def make_candidate_record(**overrides) -> dict:
    record = {
        "age": 39, "booking_lead_days": 12, "previous_appointments": 2, "previous_no_shows": 0,
        "distance_to_clinic_km": 19.3, "gender": "Female", "appointment_type": "Follow-up",
        "appointment_day": "Tuesday", "appointment_time": "Afternoon", "reminder_channel": "WhatsApp",
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Default behavior — must be unaffected by the Week 6 addition
# ---------------------------------------------------------------------------

def test_predict_defaults_to_baseline_when_model_name_omitted():
    result = predict(make_baseline_record())
    assert result["model_used"] == "baseline"


def test_predict_baseline_explicit_matches_default():
    record = make_baseline_record()
    default_result = predict(record)
    explicit_result = predict(record, model_name="baseline")
    assert default_result == explicit_result


# ---------------------------------------------------------------------------
# Routing correctness — dispatcher output must match calling the underlying
# implementation directly (proves the dispatcher adds no side effects)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["logistic_regression", "gradient_boosting"])
def test_predict_routes_to_candidate_model_correctly(model_name):
    record = make_candidate_record()
    via_dispatcher = predict(record, model_name=model_name)
    via_direct_call = predict_with_candidate(record, model_name=model_name)
    assert via_dispatcher == via_direct_call


@pytest.mark.parametrize("model_name", ["logistic_regression", "gradient_boosting"])
def test_predict_candidate_output_has_model_used_field(model_name):
    result = predict(make_candidate_record(), model_name=model_name)
    assert result["model_used"] == model_name


# ---------------------------------------------------------------------------
# Error routing — each model_name's own validation errors surface correctly
# ---------------------------------------------------------------------------

def test_predict_unknown_model_name_raises_value_error():
    with pytest.raises(ValueError):
        predict(make_candidate_record(), model_name="random_forest")


def test_predict_baseline_missing_field_raises_prediction_input_error():
    record = make_baseline_record()
    del record["reminder_sent"]
    with pytest.raises(PredictionInputError):
        predict(record, model_name="baseline")


@pytest.mark.parametrize("model_name", ["logistic_regression", "gradient_boosting"])
def test_predict_candidate_missing_field_raises_candidate_model_input_error(model_name):
    record = make_candidate_record()
    del record["distance_to_clinic_km"]
    with pytest.raises(CandidateModelInputError):
        predict(record, model_name=model_name)


@pytest.mark.parametrize(
    "model_name", ["baseline", "logistic_regression", "gradient_boosting"]
)
def test_predict_rejects_forbidden_field_regardless_of_model(model_name):
    record = (
        make_baseline_record(waiting_time_minutes=20)
        if model_name == "baseline"
        else make_candidate_record(waiting_time_minutes=20)
    )
    with pytest.raises((PredictionInputError, CandidateModelInputError)):
        predict(record, model_name=model_name)
