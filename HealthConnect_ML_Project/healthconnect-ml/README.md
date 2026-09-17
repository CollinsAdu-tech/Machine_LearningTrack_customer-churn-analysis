# HealthConnect-ML

Machine Learning Engineering track — AnalystLab Africa Experience Lab.
ML pipeline for the HealthConnect Appointment No-Show Prediction System.

## Project Overview

HealthConnect Clinic wants to predict which scheduled appointments are at
risk of a no-show, so staff can prioritize reminders and interventions.
This repository implements the full ML engineering pipeline — ingestion,
validation, cleaning, feature engineering, preprocessing, model training,
evaluation, and prediction — built and verified against handoffs from the
Data Science track across Weeks 5 and 6.

## ML Problem

Supervised binary classification: predict whether a scheduled appointment
will result in **No-Show (1)** or **Attended (0)**. `Cancelled`
appointments are excluded from the binary target (pending business/PM
confirmation — see `docs/data_dictionary_notes.md`).

## Available models

Three models are callable through one interface —
`predict(record, model_name=...)`:

| `model_name` | Source | Feature count | Notes |
|---|---|---|---|
| `"baseline"` (default) | Week 5, trained by this repo | 16 | Original pipeline, unchanged since Week 5 |
| `"logistic_regression"` | Week 6, Data Science artifact | 11 | Verified reload-identical, metrics independently reproduced |
| `"gradient_boosting"` | Week 6, Data Science artifact | 11 | DS's recommended candidate — see caveats in `reports/week6_model_verification.md` |

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
├── tests/                  67 tests across 7 files
├── models/
│   ├── model_v1.joblib        Week 5 baseline
│   └── candidates/            Week 6 Data Science artifacts (LR + GB)
├── configs/                config.yaml — single source of truth for both
│                           the Week 5 baseline and Week 6 candidates
├── reports/                data quality, pipeline docs, Week 6 integration
│                           evidence, issue log, DS feedback, Week 7 plan
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

# Predict using the Week 5 baseline (default)
python -c "from src.models.predict import predict; print(predict({...}))"

# Predict using a Week 6 Data Science candidate model
python -c "
from src.models.predict import predict
record = {
    'age': 39, 'booking_lead_days': 12, 'previous_appointments': 2,
    'previous_no_shows': 0, 'distance_to_clinic_km': 19.3, 'gender': 'Female',
    'appointment_type': 'Follow-up', 'appointment_day': 'Tuesday',
    'appointment_time': 'Afternoon', 'reminder_channel': 'WhatsApp',
}
print(predict(record, model_name='gradient_boosting'))
"
```

## Testing

```bash
pytest tests/ -v
```

**67/67 tests passing** across 7 files:
- `test_data.py`, `test_preprocessing.py`, `test_features.py`,
  `test_model.py`, `test_pipeline.py` — Week 5 baseline (33 tests,
  unchanged since Week 5)
- `test_candidate_models.py` — Week 6 Data Science model integration (21
  tests), including a pinned regression test for a confirmed
  `reminder_channel` null-handling bug found during integration
- `test_predict_dispatcher.py` — Week 6 model-routing layer (13 tests)

Verified reproducible from a genuinely clean environment — see
`reports/week6_reproducibility_check.md`.

## Model Evaluation

**Week 5 baseline** (held-out test set n=948, plain stratified split):

| Metric | Value |
|---|---|
| Recall (No-Show) — primary | 0.676 |
| Accuracy | 0.615 |
| ROC-AUC | 0.667 |

**Week 6 Data Science candidates** (held-out test set n=966,
`GroupShuffleSplit` grouped by `patient_id`, independently reproduced):

| Metric | Logistic Regression | Gradient Boosting |
|---|---|---|
| Accuracy | 63.46% | 65.11% |
| ROC-AUC | 0.6825 | 0.6941 |

**Note:** the baseline and candidate numbers are not directly comparable
— different split methodology. Flagged in `reports/week6_issue_log.md`
and `reports/week7_ml_engineering_testing_plan.md` as a Week 7 fix.

Full details: `docs/model_card.md`, `reports/week6_model_verification.md`.

## Documentation

- `docs/architecture.md` — system + repository architecture
- `docs/data_dictionary_notes.md` — per-column ML treatment notes
- `docs/model_card.md` — model details, evaluation, limitations
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

## Known open items (not silently carried forward)

- Two conditional-feature assumptions from Week 5 remain unresolved
  (reminder timing, historical-count construction) — pending PM/business
  input, per Data Science's own handoff.
- Week 5 baseline still uses a plain stratified split, not group-aware —
  makes it not directly comparable to the Week 6 candidates. Week 7 fix.
- Model selection (baseline vs. LR vs. GB) is intentionally undecided —
  Data Science's own caution against treating the GB improvement as
  settled (small effect, single-seed) means this is a Week 7+
  business/PM decision, not made here.

## Future Improvements

- Group-aware train/test split for the Week 5 baseline (Week 7).
- Schema-drift detection for the candidate model artifacts (Week 7).
- Runtime/performance benchmarking (Week 7).
- FastAPI-based prediction service.
- Docker containerization and cloud deployment.
- Automated retraining and drift monitoring per the Week 4 architecture.

## License

MIT — see `LICENSE`.
