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
| `src/features/feature_engineering.py` | The 6 engineered features from the Data Science handoff |
| `src/models/model_utils.py` | Feature selection + the ColumnTransformer (impute/scale/encode) |
| `src/models/train.py` | Split + fit the Logistic Regression baseline; save artifacts |
| `src/models/evaluate.py` | Recall-first metric computation |
| `src/models/predict.py` | Input/output contract + leakage/missing-field rejection |
| `src/pipeline/pipeline.py` | Sequences all of the above behind one entry point |

## Data flow

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

Every stage traces to a specific section of the Week 5 Data Science ->
ML Engineering handoff — see `reports/ml_pipeline_documentation.md`.
