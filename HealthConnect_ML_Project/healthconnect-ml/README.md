# HealthConnect-ML

Machine Learning Engineering track — AnalystLab Africa Experience Lab.
ML pipeline for the HealthConnect Appointment No-Show Prediction System.

**Status: Week 8 — Final Integration & Presentation.** Final model
confirmed: **Gradient Boosting**. See `docs/model_card.md` for the full
decision, including an important caveat about what "final" does and
doesn't mean here.

## Project Overview

HealthConnect Clinic wants to predict which scheduled appointments are at
risk of a no-show, so staff can prioritize reminders and interventions.
This repository implements the full ML engineering pipeline — ingestion,
validation, cleaning, feature engineering, preprocessing, model training,
evaluation, and prediction — built and verified against handoffs from the
Data Science track across Weeks 5 through 8.

## ML Problem

Supervised binary classification: predict whether a scheduled appointment
will result in **No-Show (1)** or **Attended (0)**. `Cancelled`
appointments are excluded from the binary target. This exclusion, along
with two related modeling assumptions, is documented as an explicit
assumption rather than an externally confirmed fact — see
`docs/data_dictionary_notes.md` and the Limitations section of
`docs/model_card.md`.

## Available models

Three models are callable through one interface —
`predict(record, model_name=...)`. A convenience `predict_final(record)`
always uses the confirmed final model.

| `model_name` | Source | Feature count | Status |
|---|---|---|---|
| `"baseline"` (default for `predict()`) | Week 5, trained by this repo | 16 | Original pipeline, unchanged since Week 5 |
| `"logistic_regression"` | Week 6, Data Science artifact | 11 | Verified reload-identical, metrics independently reproduced; kept for comparison |
| **`"gradient_boosting"`** | Week 6, Data Science artifact | 11 | **⭐ Confirmed final model (Week 8)** — see `docs/model_card.md` for the full reasoning, including the "beats LR on available data, not proven better as an algorithm in general" distinction |

```python
from src.models.predict import predict, predict_final

predict(record)                                  # baseline (predict()'s own default)
predict(record, model_name="gradient_boosting")  # explicit final model
predict_final(record)                            # same as above, via config.FINAL_MODEL_NAME
```

## Repository Structure

```
healthconnect-ml/
├── data/                   raw/ (original CSV) and processed/ (derived)
├── notebooks/              4 executed, documented notebooks (Week 5)
├── src/
│   ├── data/                 load_data, validate_data, preprocess
│   ├── features/             feature_engineering
│   ├── models/                model_utils, train, evaluate, predict,
│   │                          candidate_models (Week 6)
│   └── pipeline/               end-to-end orchestration
├── tests/                  88 tests across 8 files
├── models/
│   ├── model_v1.joblib        Week 5 baseline
│   └── candidates/            Week 6 Data Science artifacts (LR + final GB)
├── configs/                config.yaml — single source of truth, including
│                           the Week 8 `final_model` selection
├── reports/                data quality, pipeline docs, Week 6-8 integration
│                           evidence, issue logs, DS feedback, final decision
└── docs/                   architecture, data dictionary notes, model card
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies are version-pinned as of Week 6 — see
`reports/week6_reproducibility_check.md` for the fresh-environment
verification behind this.

## Usage

```bash
# Run the entire Week 5 pipeline end-to-end (ingest -> validate -> clean ->
# engineer -> split -> train -> evaluate -> save)
python -m src.pipeline.pipeline

# Predict using the Week 5 baseline (predict()'s own default)
python -c "from src.models.predict import predict; print(predict({...}))"

# Predict using the confirmed FINAL model (Gradient Boosting)
python -c "
from src.models.predict import predict_final
record = {
    'age': 39, 'booking_lead_days': 12, 'previous_appointments': 2,
    'previous_no_shows': 0, 'distance_to_clinic_km': 19.3, 'gender': 'Female',
    'appointment_type': 'Follow-up', 'appointment_day': 'Tuesday',
    'appointment_time': 'Afternoon', 'reminder_channel': 'WhatsApp',
}
print(predict_final(record))
"
```

## Testing

```bash
pytest tests/ -v
```

**88/88 tests passing** across 8 files:
- `test_data.py`, `test_preprocessing.py`, `test_features.py`,
  `test_model.py`, `test_pipeline.py` — Week 5 baseline (33 tests,
  unchanged since Week 5)
- `test_candidate_models.py` — Week 6 Data Science model integration (21
  tests), including a pinned regression test for a confirmed
  `reminder_channel` null-handling bug found during integration
- `test_predict_dispatcher.py` — Week 6/8 model-routing layer (16 tests):
  routing correctness, error-type routing, and the Week 8
  `predict_final()` convenience wrapper
- `test_week7_edge_cases.py` — Week 7 invalid-input testing (18 tests):
  found and fixed 3 real bugs (null-valued required fields, wrong-type
  values, a logically-impossible field combination silently reaching the
  model), and documented 3 known, lower-severity input gaps — see
  `reports/week7_pipeline_test_results.md` for the full Test → Finding →
  Action → Retest record.

Verified reproducible from a genuinely clean environment — see
`reports/week6_reproducibility_check.md`.

## Model Evaluation

**Final model — Gradient Boosting** (held-out test set n=966,
`GroupShuffleSplit` grouped by `patient_id`):

| Metric | Value |
|---|---:|
| Accuracy | 65.11% |
| ROC-AUC | 0.6941 |
| Recall (No-Show) | 66.7% |
| Confusion matrix | TN 307 / FP 176 / FN 161 / TP 322 |

**Important:** 0.6941 is the number for this specific trained artifact —
independently confirmed to reliably beat LR on the available data, but
this margin is **not guaranteed to survive a retrain** on different data
(a retrained comparison ranged 0.648–0.704 across 5 seeds). Full reasoning
in `docs/model_card.md`.

**Week 5 baseline** (held-out test set n=948, plain stratified split —
not directly comparable to the above due to different split methodology):

| Metric | Value |
|---|---|
| Recall (No-Show) — primary | 0.676 |
| Accuracy | 0.615 |
| ROC-AUC | 0.667 |

Full details: `docs/model_card.md`, `reports/week8_final_model_decision.md`.

## Documentation

- `docs/architecture.md` — system + repository architecture
- `docs/data_dictionary_notes.md` — per-column ML treatment notes
- `docs/model_card.md` — **final model card**, decision reasoning, evaluation, limitations
- `docs/week5_implementation_summary.md` — Week 5 summary
- `reports/data_quality_report.md` — validation report on the raw data
- `reports/ml_pipeline_documentation.md` — Week 5 pipeline documentation
- `reports/week6_model_verification.md` — full Week 6 artifact verification evidence
- `reports/week6_issue_log.md` — issues found/resolved during Week 6 integration
- `reports/week6_ds_feedback.md` — feedback sent back to Data Science
- `reports/week6_cross_track_integration.md` — formal cross-track integration record
- `reports/week6_reproducibility_check.md` — clean-environment verification
- `reports/week6_project_summary.md` — concise Week 6 summary
- `reports/week7_ml_engineering_testing_plan.md` — proposed Week 7 focus
- `reports/week7_ds_handoff_update.md` — Week 7 Data Science update (multi-seed retraction, segment weak spots)
- `reports/week7_pipeline_test_results.md` — Week 7 edge-case testing results (3 bugs found and fixed, 3 gaps documented)
- `reports/week7_multiseed_verification.md` — independent re-test that found the Week 7 multi-seed discrepancy
- `reports/week7_ds_feedback.md` — the clarifying question sent to Data Science about that discrepancy
- `reports/week7_cross_track_testing.md` — mandatory Week 7 HC-POD cross-track testing record
- `reports/week7_project_summary.md` — concise Week 7 summary
- `reports/week8_multiseed_resolution.md` — how the Week 7 discrepancy was resolved and independently confirmed
- `reports/week8_final_model_decision.md` — the final model decision, in full

## Project Progress

- **Week 4:** ML system design, architecture, and repository structure defined.
- **Week 5:** Full pipeline implemented end-to-end against a verified
  Data Science handoff; restructured into a modular `data/features/
  models/pipeline` package layout with YAML-based configuration.
- **Week 6:** Integrated two Data Science candidate models (Logistic
  Regression, Gradient Boosting) via a new `candidate_models.py` module
  and an extended `predict()` dispatcher. Independently reproduced all of
  Data Science's reported evaluation metrics. Found and fixed a real
  integration bug (`reminder_channel` null handling), reported it back to
  Data Science, and pinned it with a regression test. Verified full
  reproducibility from a clean environment with pinned dependencies.
- **Week 7:** Data Science reported a multi-seed test retracting the
  Week 6 "GB beats LR" claim. Independently re-tested against the actual
  delivered artifacts — did not reproduce (this pipeline: GB won 5/5,
  mean +0.0121, vs. their reported 2/5, mean −0.002). Traced to a likely
  methodological difference and sent back a specific clarifying question
  rather than adopting either conclusion. Separately, a full edge-case
  testing pass found and fixed 3 real input-validation bugs.
- **Week 8:** Data Science confirmed the methodology difference (they
  retrained both models per seed; this pipeline's re-test evaluated the
  same fixed artifacts across test-set resamples) and independently
  demonstrated GB winning 5/5 on resampling without retraining — **this
  pipeline reproduced that demonstration independently and it matched.**
  **Gradient Boosting confirmed as the final model**, with an explicit,
  narrower claim ("beats LR on available data," not "better as an
  algorithm generally") carried through to the model card. The two
  previously-open conditional-feature assumptions were closed as
  documented assumptions (not independently confirmed facts, since no
  live business stakeholder exists in this exercise). Added
  `predict_final()` as a convenience wrapper for demo/presentation use.

## Known open items (not silently carried forward)

- **The final model's advantage is data-specific, not algorithm-general
  — stated explicitly, not smoothed over.** Gradient Boosting reliably
  beats Logistic Regression on the actual available data (independently
  confirmed twice — once via direct multi-seed resampling, once via
  reproducing Data Science's own resampling demonstration). It is *not*
  established that GB would reliably win if the whole pipeline were
  retrained on different data. Both models remain available via
  `model_name` for this reason.
- The two conditional-feature assumptions (reminder timing,
  historical-count construction) are now **documented assumptions**,
  not confirmed facts — there is no live business/PM stakeholder in this
  exercise to formally verify them. See `docs/model_card.md`.
- Week 5 baseline still uses a plain stratified split, not group-aware —
  not directly comparable to the final model's metrics. Not fixed, and
  not required, since the baseline was not selected as final.
- Two segments (age 65+, Specialist Consultation) show meaningfully lower
  reliability for the final model — no confidence-flagging logic has been
  added to `predict()`'s output yet.

## Future Improvements

- Segment-aware confidence flagging (age 65+, Specialist Consultation).
- Group-aware train/test split for the Week 5 baseline, if it's ever
  reconsidered.
- Schema-drift detection for the candidate model artifacts.
- Runtime/performance benchmarking.
- Calibration testing for the final model (not yet performed).
- Fairness/bias evaluation beyond the two segments already checked.
- FastAPI-based prediction service.
- Docker containerization and cloud deployment.
- Automated retraining and drift monitoring per the Week 4 architecture.
- Validation on real (non-synthetic) data before any production use.

## License

MIT — see `LICENSE`.
