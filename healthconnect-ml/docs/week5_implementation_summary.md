# Week 5 Implementation Summary — Machine Learning Engineering

## What was planned (Week 4)
Design the ML system architecture, define the ML workflow, and propose a
repository structure for the HealthConnect no-show prediction system.

## What was completed (Week 5)
Received and independently verified the Data Science → ML Engineering
handoff, then built and tested the full pipeline against it:

1. **Verification step** — 6 key handoff claims (leakage in
   `waiting_time_minutes`, structural nulls in `reminder_channel`, patient
   overlap, date/weekday consistency, target class split) were re-checked
   directly against the raw CSV. All matched exactly.
2. **Data layer** (`src/data/`) — ingestion, schema/quality validation,
   cleaning (target filtering/encoding, structural-null handling, leakage
   removal).
3. **Feature layer** (`src/features/`) — the 6 engineered features from the
   handoff.
4. **Model layer** (`src/models/`) — shared preprocessing utilities, model
   training (Logistic Regression baseline), evaluation (Recall-first),
   and a prediction module with input-contract enforcement.
5. **Pipeline layer** (`src/pipeline/`) — a single orchestrator sequencing
   every stage end-to-end.
6. **Configuration** — moved from inline Python constants to
   `configs/config.yaml`, with a thin loader in `src/config.py`, so
   thresholds and feature lists are editable without touching pipeline code.
7. **Testing** — 33 unit/integration tests across `tests/test_data.py`,
   `test_preprocessing.py`, `test_features.py`, `test_model.py`, and
   `test_pipeline.py`. All passing.

## Baseline results (held-out test set, n=948)
Recall 0.676, Precision 0.612, F1 0.643, ROC-AUC 0.667, Accuracy 0.615.
See `docs/model_card.md` for full details.

## Key decisions and why
- `Cancelled` appointments excluded from the binary target — flagged as
  pending business/PM confirmation, not silently assumed final.
- `waiting_time_minutes` dropped entirely — confirmed data leakage,
  verified independently against the raw data before removal.
- Two feature groups (reminder timing, historical-count construction) kept
  in the baseline but marked conditional in `configs/config.yaml`.

## Challenges encountered
- `distance_bucket` is computed before distance imputation, so the 90 rows
  with missing distance get a `"nan"` bucket category — functional but
  flagged for Week 6 cleanup.
- Group-aware (`patient_id`) train/test splitting is not yet implemented —
  `patient_id` is dropped during cleaning before the split step.

## Cross-track collaboration
Received a full Data Science → ML Engineering handoff this week (target
definition, feature list, leakage findings, preprocessing spec, baseline
model choice, evaluation priorities). Every configuration decision in this
pipeline traces to a specific section of that handoff.

## Proposed focus for Week 6
1. Group-aware train/test split on `patient_id`.
2. Compare Random Forest / Gradient Boosting against the Logistic
   Regression baseline, reusing the same preprocessing pipeline.
3. Resolve the two open conditional-feature questions with Project
   Management/business, then re-run and compare.
4. Fix `distance_bucket` to compute after imputation.
5. Fairness check: evaluate metrics separately across `gender`/`age_group`.
