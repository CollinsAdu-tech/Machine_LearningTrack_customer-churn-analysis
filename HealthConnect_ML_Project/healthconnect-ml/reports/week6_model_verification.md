# Week 6 — Data Science Model Verification & Integration

## Purpose

This document records the evidence behind every claim made in
`configs/config.yaml`'s `candidate_models` section and
`src/models/candidate_models.py`. Per the engineering standard for this
project, no result below is asserted without an actual executed test —
each row states what was run and what it returned.

## Artifacts received

- `healthconnect_logreg_pipeline.joblib`
- `healthconnect_gb_pipeline.joblib`
- `HealthConnect_Week6_DataScience_Deliverable.md`

## Verification performed

| # | Claim (from DS handoff) | Method | Result |
|---|---|---|---|
| 1 | Both are full `sklearn.Pipeline` objects (preprocessing + model) | `type()` + structural inspection | Confirmed — `ColumnTransformer` (`num`: impute+scale, `cat`: one-hot) → `LogisticRegression` / `GradientBoostingClassifier` |
| 2 | GB hyperparameters: `n_estimators=100, max_depth=2, learning_rate=0.05` | Inspected fitted estimator | Confirmed exactly |
| 3 | One-hot category order identical across both models | Read `cat_encoder.categories_` on both | Confirmed — identical on both, matches handoff exactly |
| 4 | Reload → identical predictions | Saved + reloaded both artifacts independently in this environment, compared `predict_proba()` on 50 real rows | Confirmed identical (`np.allclose` = True) for both models |
| 5 | Works on raw DataFrame, no separate encoding/scaling needed | Passed a raw row with missing `distance_to_clinic_km` through the full pipeline | Confirmed — bundled `SimpleImputer` correctly imputes the fitted median (8.6 km) |
| 6 | Works on raw DataFrame — **no other preprocessing needed at all** | Passed a raw row with a **null `reminder_channel`** through the full pipeline | **False — see "Confirmed Bug" below** |
| 7 | `handle_unknown='ignore'` degrades gracefully on unseen category | Passed `appointment_type="Telehealth Video Call"` (not a fitted category) | Confirmed — no error, valid probability returned |
| 8 | Reported evaluation metrics (Accuracy, ROC-AUC, confusion matrix) for both models | Rebuilt `GroupShuffleSplit(test_size=0.2, random_state=42)` grouped by `patient_id` from scratch, re-scored both models on the resulting test set | **Exact match on every number, both models** (see table below) |
| 9 | 0 patient overlap between train/test | Set-intersection check on `patient_id` in my own reproduction of their split | Confirmed: 0 |

### Reproduced evaluation metrics

| Metric | Logistic Regression (reported / reproduced) | Gradient Boosting (reported / reproduced) |
|---|---|---|
| Accuracy | 63.46% / **63.46%** | 65.11% / **65.11%** |
| ROC-AUC | 0.6825 / **0.6825** | 0.6941 / **0.6941** |
| Confusion matrix | TN294 FP189 FN164 TP319 / **identical** | TN307 FP176 FN161 TP322 / **identical** |

Every figure in Section 5 of the DS deliverable is independently
reproducible from the raw CSV and the artifact alone — no part of the
handoff was taken on trust without a check.

## Confirmed Bug: `reminder_channel` null handling

**Claim tested:** DS's message stated their pipelines work directly on a
raw DataFrame with "no separate encoding/scaling step needed on their
end." This is true for encoding and scaling. It is **not** true for one
specific null-handling case.

**Mechanism:** When pandas reads the raw CSV, a missing `reminder_channel`
value becomes `NaN` inside a `str`-dtype column. The fitted
`OneHotEncoder(handle_unknown='ignore')` does not recognize `NaN` as any
of its four fitted categories (`Email`, `None`, `SMS`, `WhatsApp`). It
silently produces an all-zero one-hot row for that record — behaviorally
identical to how it would treat a genuinely novel, never-before-seen
category. It does **not** raise an error, and it does **not** match the
fitted `"None"` category, even though that category exists specifically
to represent "no reminder was sent."

**Evidence (executed, not asserted):**

```
Proba with raw NaN reminder_channel  (GB):     0.775913694359012
Proba with explicit "None" string    (GB):     0.7891248034001417
IDENTICAL? False

Proba with raw NaN reminder_channel  (LR):     0.7652819263462073
Proba with explicit "None" string    (LR):     0.7936890967059312
Same bug present in LR pipeline: True
```

**Scope:** 1,366 of 5,000 rows (27.32%) in the raw dataset have a null
`reminder_channel`. Left unaddressed, every one of those rows would
receive a silently degraded prediction on any live system built on these
artifacts.

**Fix:** `src/models/candidate_models.py::prepare_candidate_input()` is
the mandatory gate that converts `reminder_channel` nulls to the literal
string `"None"` before either pipeline is called. This is the same
transformation already implemented in the Week 5 pipeline
(`src/data/preprocess.py::handle_structural_nulls`) for the baseline model
— reused conceptually, reimplemented locally in `candidate_models.py` to
keep this module's dependency footprint self-contained.

**Regression test:** `tests/test_candidate_models.py::test_raw_nan_reminder_channel_differs_from_prepared_input`
runs this exact comparison against both live artifacts on every test run,
parametrized across both models. This directly satisfies DS's own Week 7
testing requirement #6 ("regression test on leakage controls... carried
over from Week 5, still unimplemented as an automated test").

**Contrast case, explicitly verified NOT to need a fix:**
`distance_to_clinic_km`'s raw `NaN` values are correctly handled by the
bundled `SimpleImputer(strategy="median")` — verified the imputed value
matches the fitted median (8.6 km) exactly. No preparation-gate
intervention needed for this column.

## What changed in the repository

| File | Change |
|---|---|
| `configs/config.yaml` | Added `candidate_models:` section — separate schema, feature-derivation spec, forbidden columns, fitted categories, reported evaluation numbers. Week 5 baseline config (`model:`, `features:`, etc.) untouched. |
| `src/config.py` | Added accessors for the new `candidate_models` config block. Existing accessors untouched. |
| `src/models/candidate_models.py` | **New file.** Loads both DS artifacts, implements the mandatory input-preparation gate, provides `predict_with_candidate()`. |
| `src/models/predict.py` | **Updated.** `predict()` is now a dispatcher (`model_name="baseline"` by default). Original baseline logic moved verbatim into `_predict_baseline()` — not modified, only renamed and relocated. |
| `models/candidates/` | **New directory.** Both `.joblib` artifacts stored here, separate from `models/model_v1.joblib` (the Week 5 baseline). |
| `tests/test_candidate_models.py` | **New file.** 21 tests: artifact loading, schema verification, the pinned bug regression test (both models), input/output contract validation, reload-identity, and independently-reproduced evaluation metrics. |
| `tests/test_predict_dispatcher.py` | **New file.** 13 tests: default-to-baseline behavior, routing correctness (dispatcher output == direct call output), and error-type routing per model. |

**What was explicitly NOT changed:** `src/data/`, `src/features/feature_engineering.py`,
`src/models/{train,evaluate,model_utils}.py`, and the Week 5
baseline model artifact. `src/models/predict.py`'s baseline logic is
unchanged in behavior (moved, not edited) — see `_predict_baseline()`.
These remain exactly as they were — the DS candidate models are
integrated as a parallel, self-contained path, not a replacement of the
existing baseline infrastructure. Full test suite confirms this: 67/67
tests pass (33 original + 21 candidate-model + 13 dispatcher, all new).

## Still open (not resolved by this integration step)

1. **Two conditional assumptions inherited from Week 5**, restated by DS as
   still unresolved: whether `reminder_sent`/`reminder_channel` are
   legitimately available at the intended prediction time, and whether
   `previous_appointments`/`previous_no_shows` exclude the current
   appointment. Neither candidate model resolves these — they're baked
   into the training data DS used, same as the Week 5 baseline.
2. **Model selection** — this integration makes both models callable
   through the same interface (`predict_with_candidate(record, model_name)`).
   It does not choose one as "the" production model. That's a decision for
   Project Management / business, informed by DS's explicit caution that
   the GB improvement is small and single-seed (their own Week 7 item #1).
3. **`src/models/predict.py`** has been updated with a routing layer:
   `predict(record, model_name="baseline")` now dispatches to either the
   unchanged Week 5 baseline path or `predict_with_candidate()` for
   `"logistic_regression"`/`"gradient_boosting"`. `model_name` defaults to
   `"baseline"` specifically so every pre-Week-6 call site is unaffected —
   verified: all 33 original tests pass unmodified, plus 13 new dispatcher
   tests confirming routing correctness (dispatcher output is checked
   equal to calling `predict_with_candidate()` directly, proving the
   dispatcher adds no side effects) and that each model's own validation
   errors (`PredictionInputError` vs `CandidateModelInputError`) surface
   correctly through the shared entry point.

**Full test suite after this step: 67/67 passing** (33 baseline + 21
candidate-model + 13 dispatcher).
