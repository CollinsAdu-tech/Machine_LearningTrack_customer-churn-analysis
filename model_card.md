# HealthConnect No-Show Prediction — Model Card (Baseline v1)

## Model Details
- **Model type:** Logistic Regression (scikit-learn)
- **Version:** v1 (Week 5 baseline)
- **Hyperparameters:** `C=1.0`, `class_weight=None`, `max_iter=1000`, `random_state=42`
- **Artifact:** `models/model_v1.joblib` (+ `models/model_metadata.json`)

## Intended Use
Decision-support tool for HealthConnect staff to flag appointments at
elevated risk of a no-show, so reminder/intervention effort can be
prioritized. **Not** intended as an automatic system for denying or
altering care, and not validated on real patient data (see Limitations).

## Training Data
- `HealthConnect_Appointment_Data.csv` — 5,000 fictional, anonymized
  appointment records.
- After cleaning (dropping `Cancelled`, ~5.3%): 4,737 rows.
- Split: 80/20 stratified, `random_state=42` → 3,789 train / 948 test.

## Features
16 features — 6 numerical, 7 categorical, 3 boolean. Full list in
`configs/config.yaml` under `features:`. Two feature groups are flagged
conditional (see `docs/data_dictionary_notes.md`).

`waiting_time_minutes` is explicitly excluded (confirmed data leakage —
populated even for appointments the patient never attended).

## Evaluation (held-out test set, n=948)

| Metric | Value |
|---|---|
| Recall (No-Show) — primary | 0.676 |
| Precision (No-Show) | 0.612 |
| F1-score | 0.643 |
| ROC-AUC | 0.667 |
| Accuracy | 0.615 |

Confusion matrix: TN=255, FP=208, FN=157, TP=328.

Recall is the primary metric: a false negative (predicted Attended, patient
actually no-shows) reproduces the exact problem the system exists to
address — an unflagged wasted appointment slot.

## Limitations
- Trained on **synthetic/anonymized** data — performance on real
  HealthConnect patients is unverified.
- ~5,000 rows is modest for a robust baseline; class balance is close to
  even (51.2%/48.8%) so no resampling was applied.
- Train/test split is a plain stratified split, **not** group-aware —
  1,394 of 1,696 patients appear in more than one row, so the same patient
  can appear in both folds (flagged for Week 6: `GroupShuffleSplit`).
- Two feature groups (reminder timing, historical-count construction) are
  included pending confirmation of modeling assumptions not stated in the
  source data dictionary.
- No causal claims — all relationships are correlational.

## Ethical Considerations
- Model output is a probability + risk category, framed as decision
  support, not an automated action.
- No personally identifying information is used as a feature.
- Performance across demographic groups (e.g. `gender`, `age`) has not yet
  been separately evaluated — flagged as a Week 6 fairness-check item.
