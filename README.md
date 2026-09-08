# HealthConnect-ML

Machine Learning Engineering track — AnalystLab Africa Experience Lab.
ML pipeline for the HealthConnect Appointment No-Show Prediction System.

## Project Overview

HealthConnect Clinic wants to predict which scheduled appointments are at
risk of a no-show, so staff can prioritize reminders and interventions.
This repository implements the full ML engineering pipeline — ingestion,
validation, cleaning, feature engineering, preprocessing, baseline model
training, evaluation, and prediction — built directly against a verified
Data Science → ML Engineering handoff.

## ML Problem

Supervised binary classification: predict whether a scheduled appointment
will result in **No-Show (1)** or **Attended (0)**. `Cancelled`
appointments are excluded from the binary target (pending business/PM
confirmation — see `docs/data_dictionary_notes.md`).

## Repository Structure

```
healthconnect-ml/
├── data/                   raw/ (original CSV) and processed/ (derived)
├── notebooks/              4 executed, documented notebooks
├── src/
│   ├── data/                 load_data, validate_data, preprocess
│   ├── features/             feature_engineering
│   ├── models/                model_utils, train, evaluate, predict
│   └── pipeline/               end-to-end orchestration
├── tests/                  one test file per src/ area (33 tests)
├── models/                 model_v1.joblib + model_metadata.json
├── configs/                config.yaml — single source of truth
├── reports/                data quality + pipeline docs, figures, results
└── docs/                   architecture, data dictionary notes, model card
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run the entire pipeline end-to-end (ingest -> validate -> clean ->
# engineer -> split -> train -> evaluate -> save)
python -m src.pipeline.pipeline

# Or run individual stages
python -m src.data.load_data          # (via Python import, no __main__)
python -m src.models.train            # train + save the baseline model
python -m src.models.evaluate         # evaluate the saved/trained model

# Predict on a single record
python -c "from src.models.predict import predict; print(predict({...}))"
```

## Testing

```bash
pytest tests/ -v
```

33/33 tests passing: ingestion/validation, cleaning, feature engineering,
preprocessing (fit-on-train-only proof), end-to-end training, prediction
input rejection, model save/reload identity, and full pipeline
orchestration.

## Model Evaluation (Week 5 baseline, held-out test set n=948)

| Metric | Value |
|---|---|
| Recall (No-Show) — primary | 0.676 |
| Precision (No-Show) | 0.612 |
| F1-score | 0.643 |
| ROC-AUC | 0.667 |
| Accuracy | 0.615 |

Full details in `docs/model_card.md`.

## Documentation

- `docs/architecture.md` — system + repository architecture
- `docs/data_dictionary_notes.md` — per-column ML treatment notes
- `docs/model_card.md` — model details, evaluation, limitations
- `docs/week5_implementation_summary.md` — Week 5 summary
- `reports/data_quality_report.md` — validation report on the raw data
- `reports/ml_pipeline_documentation.md` — full pipeline documentation

## Project Progress

- **Week 4:** ML system design, architecture, and repository structure defined.
- **Week 5:** Full pipeline implemented end-to-end against a verified
  Data Science handoff; restructured into a modular `data/features/
  models/pipeline` package layout with YAML-based configuration. Two
  conditional-feature assumptions remain open pending PM/business
  confirmation.

## Future Improvements

- Group-aware train/test split on `patient_id` (Week 6).
- Random Forest / Gradient Boosting comparison models (Week 6).
- FastAPI-based prediction service.
- Docker containerization and cloud deployment.
- Automated retraining and drift monitoring per the Week 4 architecture.

## License

MIT — see `LICENSE`.
