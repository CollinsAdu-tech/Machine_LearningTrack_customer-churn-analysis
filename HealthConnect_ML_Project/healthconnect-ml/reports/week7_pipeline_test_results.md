# Week 7 — ML Pipeline Testing & Reliability Results

Format follows the assignment's required Testing, Refinement & Validation
Record fields (Section 12). Every row below corresponds to an actual test
in `tests/test_week7_edge_cases.py`, run against all three models
(baseline, logistic_regression, gradient_boosting) unless noted otherwise.

## Component Tested
The integrated `predict()` dispatcher (`src/models/predict.py`) and its
routing to both the Week 5 baseline and the Week 6 Data Science candidate
models (`src/models/candidate_models.py`), specifically its handling of
invalid, malformed, and edge-case input — Week 7 task #9 ("Test
appropriate invalid or unexpected input scenarios").

## Testing Objective
Determine whether the pipeline correctly rejects invalid input before it
reaches a model (rather than silently mispredicting or crashing with an
unhelpful error), across all three routable models.

---

## Test Results

| # | Test/Scenario | Expected Result | Actual Result (before fix) | Pass/Fail | Issue Identified | Action Taken | Retest Result |
|---|---|---|---|---|---|---|---|
| 1 | Required field present but set to `None` (e.g. `{"age": None}`) | Rejected with a clear, named error | **Silently accepted** — passed the "key exists" check and was fed straight into the model with no warning | **FAIL** | `None` is not the same as "missing," and the existing validation only checked for the key's presence, not its value | Added `find_null_fields()` in `model_utils.py`; wired into both `predict.py` and `candidate_models.py` to reject before reaching the model | **PASS** — `test_null_required_field_is_rejected`, all 3 models |
| 2 | Wrong-type numeric field (e.g. `age="thirty-nine"`) | Rejected with a clear, named error identifying the bad field | Propagated into sklearn's `SimpleImputer`, which raised a raw `ValueError: Cannot use median strategy with non-numeric data` — no indication of which field or model caused it | **FAIL** | No type validation existed before data reached the model; error surfaced from inside sklearn internals, not our own contract | Added `find_non_numeric_fields()` in `model_utils.py`; wired into both prediction paths | **PASS** — `test_non_numeric_value_for_numeric_field_is_rejected`, all 3 models |
| 3 | Logically impossible combination: `previous_no_shows=10, previous_appointments=1` | Rejected — this exact inconsistency is already an automated data-quality check on training data (`src/data/validate_data.py`), but was not enforced on a live prediction request | Silently accepted and fed to the model, producing a prediction from an impossible input | **FAIL** | Training-time validation existed; prediction-time validation for the same logical rule did not | Added an explicit `previous_no_shows > previous_appointments` check in both `predict.py` and `candidate_models.py` | **PASS** — `test_previous_no_shows_exceeding_appointments_is_rejected`, all 3 models |
| 4 | Negative age (`age=-5`) | Undecided going in — tested to see actual behavior | Model still returns a valid-range probability (0–1); does not crash or return garbage | **PASS (as a "doesn't crash" test), but flags a real gap** | Negative age is not range-checked anywhere in the pipeline | **Not fixed this pass** — lower severity than #1–#3 since it doesn't crash or silently corrupt a *type contract*, only accepts semantically nonsensical input | Documented as a known, accepted gap — see Remaining Issues below |
| 5 | Empty string for a categorical field (`gender=""`) | Undecided going in — tested to see actual behavior | `OneHotEncoder(handle_unknown='ignore')` treats `""` exactly like any other unseen category — no crash, no special handling | **PASS (mechanically), flags a design question** | An empty string is arguably a different failure mode than "a genuinely new category we haven't seen yet" (e.g. a new appointment type), but the pipeline currently can't tell them apart | **Not fixed this pass** — needs a product decision (should `""` be rejected outright?) more than a technical one | Documented as a known, accepted gap — see Remaining Issues below |
| 6 | Implausible distance (`distance_to_clinic_km=999999`) | Undecided going in — tested to see actual behavior | Model extrapolates rather than erroring; for the baseline and LR candidate this saturates probability at exactly 1.0 | **PASS (mechanically), flags a calibration note** | No upper-bound sanity check exists for any numeric field; probability saturating at exactly 1.0 on an absurd input is a mild calibration concern worth knowing about, not a functional bug | **Not fixed this pass** | Documented as a known, accepted gap — see Remaining Issues below |

**Evidence:** `tests/test_week7_edge_cases.py` — 18 tests (6 scenarios ×
3 models), all passing after fixes. Run via `pytest
tests/test_week7_edge_cases.py -v`. Full suite: **85/85 passing**
(67 from Weeks 5–6, unchanged, + 18 new this pass).

---

## Interpretation

Tests 1–3 found genuine bugs — inputs that should never reach a model
were reaching it, either silently (tests 1, 3) or with an unhelpful raw
error (test 2). All three are fixed and permanently pinned by automated
tests, so they can't silently regress.

Tests 4–6 are different in kind: the pipeline doesn't misbehave
mechanically (no crash, output stays in valid range), but the input
itself is semantically questionable and currently gets no special
treatment. Per the assignment's own guidance ("where no issue is
identified, clearly document the successful validation rather than
inventing an issue"), these are recorded honestly as accepted current
behavior with a noted rationale — not force-fit into "bugs" to appear
more thorough, and not silently ignored either.

## Remaining Issues (carried to Week 8 consideration, not fixed here)

- **No range validation on `age`** — negative or implausibly large values
  pass through. Low severity (no crash), but worth a bounds check
  (e.g. `0 <= age <= 110`, matching the same bound already used in
  `src/data/validate_data.py` for training data).
- **Empty-string categoricals are indistinguishable from genuinely novel
  categories.** Whether this should be rejected outright is a product
  question, not purely technical — flagging for Project Management /
  Data Science input rather than deciding unilaterally.
- **No upper-bound sanity check on `distance_to_clinic_km`** (or other
  numeric fields). Related to but distinct from the calibration concern
  noted in `docs/model_card.md`'s "uncertain zone" discussion — this is
  about extreme out-of-range inputs, not the model's general uncertainty.

## Cross-track relevance

The `previous_no_shows > previous_appointments` check (test 3) enforces,
at prediction time, a data-quality rule Data Science's Week 5 handoff
established for training data (`src/data/validate_data.py`). This closes
a gap between training-time and prediction-time validation that neither
track had explicitly owned before this test surfaced it.
