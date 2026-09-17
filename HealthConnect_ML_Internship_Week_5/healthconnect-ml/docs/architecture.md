# HealthConnect ML — Architecture

## System Overview

See the Week 4 ML System Design Document (`../reports/` from the previous
submission, or the AnalystLab-provided diagrams) for the full high-level
architecture: Offline ML Development Pipeline -> Online Prediction/Serving
Pipeline -> ML Monitoring & Maintenance -> Security & Governance.

## Repository Architecture (this repo)

```
healthconnect-ml/
├── data/                  Raw and processed datasets
├── notebooks/             Exploratory / documented analysis notebooks
├── src/
│   ├── data/               Ingestion, validation, cleaning
│   ├── features/           Feature engineering
│   ├── models/             Preprocessing utils, training, evaluation, prediction
│   └── pipeline/            End-to-end orchestration
├── tests/                  Unit tests, one file per src/ area
├── models/                 Serialized model artifacts + metadata
├── configs/                config.yaml — single source of truth
├── reports/                Data-quality and pipeline documentation, results
└── docs/                   This documentation set
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `src/data/load_data.py` | Read the raw CSV; fail loudly on missing/empty file |
| `src/data/validate_data.py` | Schema + data-quality checks (Week 4 Stage 2) |
| `src/data/preprocess.py` | Target filtering/encoding, structural-null handling, leakage-column removal |
| `src/features/feature_engineering.py` | The 6 engineered features from the Week 5 Data Science handoff (baseline model only) |
| `src/models/model_utils.py` | Baseline feature selection + the ColumnTransformer (impute/scale/encode) |
| `src/models/train.py` | Split + fit the Week 5 Logistic Regression baseline; save artifacts |
| `src/models/evaluate.py` | Recall-first metric computation (baseline model) |
| `src/models/candidate_models.py` | **Week 6.** Loads the two Data Science candidate artifacts (LR, GB); mandatory input-preparation gate (`historical_no_show_rate` derivation + `reminder_channel` null fix); unified `predict_with_candidate()` |
| `src/models/predict.py` | Dispatcher: routes to the Week 5 baseline (`_predict_baseline`, unchanged) or a Week 6 candidate model (`candidate_models.predict_with_candidate`), based on `model_name` |
| `src/pipeline/pipeline.py` | Sequences the Week 5 baseline stages behind one entry point (candidate models are not yet part of this orchestrator — called directly via `predict()`) |

## Data flow — Week 5 baseline (unchanged)

```
Raw CSV
  -> load_data.load_appointment_data
  -> validate_data.validate_dataset
  -> preprocess.clean_dataset          (drop Cancelled, encode target, fix nulls, drop leakage cols)
  -> feature_engineering.engineer_features
  -> model_utils.select_model_features + train_test_split
  -> model_utils.build_preprocessor -> LogisticRegression   (sklearn Pipeline)
  -> evaluate.evaluate_model
  -> train.save_model                  (models/model_v1.joblib + metadata)
```

## Data flow — Week 6 candidate models

```
Raw record (dict)
  -> predict(record, model_name="logistic_regression" | "gradient_boosting")
  -> candidate_models._validate_request     (reject forbidden/missing fields)
  -> candidate_models.prepare_candidate_input
       -> compute_historical_no_show_rate    (derived, not a raw input)
       -> fillna(reminder_channel, "None")   (MANDATORY — see week6_model_verification.md)
  -> pipeline.predict_proba()   (DS's bundled sklearn.Pipeline: ColumnTransformer + classifier)
  -> _validate_output           (range/NaN check)
  -> {prediction, label, no_show_probability, risk_category, model_used}
```

Note the two paths use **different feature schemas** by design (16
features for the baseline, 11 for the candidates) — this reflects a
deliberate Data Science modeling decision in Week 6, not an
inconsistency to be unified. See `reports/week6_model_verification.md`
for the full reconciliation.

Every Week 5 stage traces to the Week 5 Data Science handoff — see
`reports/ml_pipeline_documentation.md`. Every Week 6 decision traces to
the Week 6 Data Science deliverable and this repo's own verification —
see `reports/week6_model_verification.md`.
