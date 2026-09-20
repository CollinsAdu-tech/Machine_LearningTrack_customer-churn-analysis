"""
Week 7 — Edge-case / invalid-input regression tests
Machine Learning Engineering Track

Covers Week 7 task #9 ("Test appropriate invalid or unexpected input
scenarios") against the fully integrated pipeline (baseline + both Week 6
candidate models). Each test below corresponds to a specific
Test -> Finding -> Action -> Retest entry in
reports/week7_pipeline_test_results.md.

Three genuine bugs were found and fixed as a direct result of writing
these tests (see find_null_fields / find_non_numeric_fields in
model_utils.py, and the previous_no_shows > previous_appointments check
added to both predict.py and candidate_models.py):
    1. A None/null value for a required field was silently accepted.
    2. A wrong-type value (string instead of number) raised a raw,
       unfriendly sklearn ValueError instead of a named input error.
    3. A logically-impossible combination (previous_no_shows >
       previous_appointments) was silently accepted and fed to the model.

Two further cases were tested and found to still pass through without
rejection. These are NOT bugs in the same sense — the pipeline doesn't
crash or silently corrupt output — but they are documented range-
validation gaps, tracked here as "known, accepted for now" rather than
left undocumented:
    4. A negative age is accepted (model still returns a valid-range
       probability, just on nonsensical input).
    5. An empty string for a categorical field is accepted (the
       OneHotEncoder treats it as an unseen category — same as any
       genuinely novel value).
    6. An implausibly large distance is accepted (the model just
       extrapolates; not a crash, but not a bounded/sane response either).

Run with:
    pytest tests/test_week7_edge_cases.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.candidate_models import CandidateModelInputError
from src.models.predict import PredictionInputError, predict

ALL_MODELS = ["baseline", "logistic_regression", "gradient_boosting"]


def make_record(model_name: str, **overrides) -> dict:
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
    if model_name == "baseline":
        record["reminder_sent"] = "Yes"
    record.update(overrides)
    return record


def _input_error_types():
    return (PredictionInputError, CandidateModelInputError)


# ---------------------------------------------------------------------------
# FOUND AND FIXED — must now be rejected with a named error, on all 3 models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_null_required_field_is_rejected(model_name):
    """
    BUG (fixed): {"age": None} used to pass the "missing field" check
    (the key is present) and reach the model silently. Now rejected.
    """
    record = make_record(model_name, age=None)
    with pytest.raises(_input_error_types()):
        predict(record, model_name=model_name)


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_non_numeric_value_for_numeric_field_is_rejected(model_name):
    """
    BUG (fixed): age="thirty-nine" used to propagate into sklearn's
    SimpleImputer and raise a raw, unfriendly ValueError. Now rejected
    with a clear, named error before it reaches the model.
    """
    record = make_record(model_name, age="thirty-nine")
    with pytest.raises(_input_error_types()):
        predict(record, model_name=model_name)


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_previous_no_shows_exceeding_appointments_is_rejected(model_name):
    """
    BUG (fixed): previous_no_shows > previous_appointments is logically
    impossible (already enforced as a data-quality check in
    src/data/validate_data.py for training data) but was NOT enforced at
    prediction time. Now rejected.
    """
    record = make_record(model_name, previous_appointments=1, previous_no_shows=10)
    with pytest.raises(_input_error_types()):
        predict(record, model_name=model_name)


# ---------------------------------------------------------------------------
# KNOWN, DOCUMENTED GAPS — currently accepted; pinned so any future
# behavior change here is a deliberate decision, not a silent regression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_negative_age_is_currently_accepted(model_name):
    """
    KNOWN GAP, not yet fixed: a negative age is not range-checked. The
    model still returns a valid-range probability (doesn't crash or
    return garbage), so this is lower severity than the three fixed bugs
    above. Tracked in reports/week7_pipeline_test_results.md as a
    candidate for Week 8, not fixed in this pass.
    """
    record = make_record(model_name, age=-5)
    result = predict(record, model_name=model_name)
    assert 0.0 <= result["no_show_probability"] <= 1.0


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_empty_string_categorical_is_currently_accepted(model_name):
    """
    KNOWN GAP, not yet fixed: an empty string for a categorical field is
    treated by OneHotEncoder as just another unseen category (same
    handle_unknown='ignore' path as a genuinely novel value). Doesn't
    crash, but arguably should be rejected as clearly invalid input
    rather than silently treated as "unknown." Tracked, not fixed here.
    """
    record = make_record(model_name, gender="")
    result = predict(record, model_name=model_name)
    assert 0.0 <= result["no_show_probability"] <= 1.0


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_extreme_distance_is_currently_accepted(model_name):
    """
    KNOWN GAP, not yet fixed: an implausible distance (999999 km) is not
    range-checked. The model extrapolates rather than erroring — for the
    baseline and LR candidate this saturates probability at exactly 1.0,
    which is itself worth a calibration note (see
    reports/week7_pipeline_test_results.md) but not a crash.
    """
    record = make_record(model_name, distance_to_clinic_km=999999)
    result = predict(record, model_name=model_name)
    assert 0.0 <= result["no_show_probability"] <= 1.0
