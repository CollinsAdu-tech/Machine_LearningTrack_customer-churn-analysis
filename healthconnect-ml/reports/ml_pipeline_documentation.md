# HealthConnect ML Pipeline — Documentation (Week 5)

## Scope

Full Week 5 ML Engineering pipeline, built directly against the Data
Science → ML Engineering handoff. Every config decision, column list, and
threshold traces back to a specific section of that handoff.

## Verifying the Data Science handoff

Before building against it, the following claims were independently
re-checked against `data/raw/HealthConnect_Appointment_Data.csv`:

| Claim | Verified |
|---|---|
| `waiting_time_minutes` populated for 2,647/2,686 No-Show+Cancelled rows | Exact match |
| `reminder_channel` nulls (1,366) match `reminder_sent == "No"`, 0 exceptions | Exact match |
| 1,696 unique `patient_id` values | Exact match |
| `booking_lead_days` matches computed date difference, 0 mismatches | Exact match |
| `appointment_day` matches computed weekday, 0 mismatches | Exact match |
| Target split after excluding Cancelled: 51.2%/48.8% | Exact match |

## Pipeline stages and modules

| Stage | Module |
|---|---|
| Data Ingestion | `src/data/load_data.py` |
| Data Validation | `src/data/validate_data.py` |
| Data Cleaning | `src/data/preprocess.py` |
| Feature Engineering | `src/features/feature_engineering.py` |
| Preprocessing (encode/scale) | `src/models/model_utils.py` |
| Model Training | `src/models/train.py` |
| Model Evaluation | `src/models/evaluate.py` |
| Prediction | `src/models/predict.py` |
| End-to-end orchestration | `src/pipeline/pipeline.py` |
| Configuration | `configs/config.yaml` (loaded via `src/config.py`) |

Run the whole pipeline in one command:
```
python -m src.pipeline.pipeline
```

## Data Cleaning (`src/data/preprocess.py`)
1. Parse `booking_date`/`appointment_date` as `%m/%d/%Y`.
2. Drop `Cancelled` rows, encode target `No-Show=1`/`Attended=0`.
   **Open item:** exclusion of Cancelled pending business/PM confirmation.
3. Fill `reminder_channel` nulls with explicit `"None"` category.
4. Drop `appointment_id`, `patient_id`, `waiting_time_minutes`, `age_group`.

Result: 5,000 → 4,737 rows.

## Feature Engineering (`src/features/feature_engineering.py`)
`prior_no_show_rate`, `is_first_time_patient`, `is_weekend_appointment`,
`lead_time_bucket`, `distance_bucket`, `reminder_sent_flag`.

**Known limitation:** `distance_bucket` computed pre-imputation — 90 rows
with missing distance get a `"nan"` string bucket. Functional, flagged for
Week 6.

## Preprocessing (`src/models/model_utils.py`)
`ColumnTransformer`, fit only on the training fold:
- Numerical: median impute → `StandardScaler`.
- Categorical: one-hot, `handle_unknown="ignore"`.
- Boolean: passthrough.

## Train/Test Split
80/20, stratified, `random_state=42`. Actual: 3,789 train / 948 test.

**Known limitation:** plain stratified split, not group-aware — 1,394 of
1,696 patients appear in more than one row. `GroupShuffleSplit` on
`patient_id` is the Week 6 upgrade.

## Baseline Model — Logistic Regression
`C=1.0`, `class_weight=None`, `max_iter=1000`, `random_state=42`.
Saved to `models/model_v1.joblib` + `models/model_metadata.json`.

## Evaluation Results (held-out test set, n=948)

| Metric | Value |
|---|---|
| **Recall (No-Show) — PRIMARY** | **0.676** |
| Precision (No-Show) | 0.612 |
| F1-score | 0.643 |
| ROC-AUC | 0.667 |
| Accuracy | 0.615 |

Confusion matrix: TN=255, FP=208, FN=157, TP=328.

157 false negatives = 157 test-set appointments that would actually
no-show, predicted as Attended — the specific failure this project exists
to reduce.

## Prediction Module (`src/models/predict.py`)
- Rejects `appointment_outcome`/`waiting_time_minutes` in requests.
- Rejects requests missing required fields, naming them.
- Returns `{prediction, label, no_show_probability, risk_category}` with
  Week 4 risk thresholds (Low ≤30%, Medium 31–60%, High 61–100%).

## Testing Evidence
33/33 tests passing across 5 files:
- `tests/test_data.py` (12) — ingestion, validation.
- `tests/test_preprocessing.py` (5) — cleaning.
- `tests/test_features.py` (5) — feature engineering.
- `tests/test_model.py` (9) — preprocessing utils, training, save/reload,
  prediction contract.
- `tests/test_pipeline.py` (2) — end-to-end orchestration.

```
pytest tests/ -v
```

## Conditional Features — Still Open
Per the handoff, two assumptions remain unconfirmed:
1. Whether `reminder_sent`/`reminder_channel` are safe given actual
   prediction timing.
2. Whether `previous_appointments`/`previous_no_shows` (and derived
   features) exclude the current appointment.

Both included in the baseline but listed separately in
`configs/config.yaml` under `conditional_features` for easy removal.

## Cross-Track Dependency Record
**Track:** Data Science → ML Engineering.
**Exchanged:** Verified dataset profile, target definition, feature/
exclusion lists, leakage findings, preprocessing spec, train/test
strategy, baseline model choice, evaluation priorities.
**Result:** Every config decision in this pipeline traces to a specific,
evidence-backed section of that handoff; open items preserved as config
flags, not silently resolved.

## Next Steps (Week 6)
1. `GroupShuffleSplit`/`GroupKFold` on `patient_id`.
2. Regression test guarding against `waiting_time_minutes` re-entering
   the feature set.
3. Random Forest / Gradient Boosting comparison models.
4. Resolve open conditional-feature questions with Project Management.
5. Fix `distance_bucket` to compute after imputation.
6. Fairness check across demographic groups.
