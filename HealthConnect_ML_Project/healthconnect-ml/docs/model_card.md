# HealthConnect No-Show Prediction — Model Card

Three models are available, callable through `predict(record, model_name=...)`.

---

## Model 1: Baseline (Week 5)

- **Model type:** Logistic Regression (scikit-learn)
- **model_name:** `"baseline"` (default)
- **Hyperparameters:** `C=1.0`, `class_weight=None`, `max_iter=1000`, `random_state=42`
- **Artifact:** `models/model_v1.joblib` (+ `models/model_metadata.json`)
- **Trained by:** this repository (`src/models/train.py`)

### Training Data
- `HealthConnect_Appointment_Data.csv` — 5,000 fictional, anonymized appointment records.
- After cleaning (dropping `Cancelled`, ~5.3%): 4,737 rows.
- Split: 80/20 **plain stratified**, `random_state=42` → 3,789 train / 948 test.
  **Not group-aware** — see Limitations.

### Features
16 features — 6 numerical, 7 categorical, 3 boolean. Full list in
`configs/config.yaml` under `features:`. Two feature groups are flagged
conditional (see `docs/data_dictionary_notes.md`).

### Evaluation (held-out test set, n=948)

| Metric | Value |
|---|---|
| Recall (No-Show) — primary | 0.676 |
| Precision (No-Show) | 0.612 |
| F1-score | 0.643 |
| ROC-AUC | 0.667 |
| Accuracy | 0.615 |

Confusion matrix: TN=255, FP=208, FN=157, TP=328.

---

## Model 2 & 3: Data Science Candidates (Week 6)

- **model_name:** `"logistic_regression"` or `"gradient_boosting"`
- **Provided by:** Data Science track (`HealthConnect_Week6_DataScience_Deliverable.md`)
- **Artifacts:** `models/candidates/healthconnect_logreg_pipeline.joblib`,
  `models/candidates/healthconnect_gb_pipeline.joblib`
- **GB hyperparameters:** `n_estimators=100, max_depth=2, learning_rate=0.05`
  (selected via `GridSearchCV` with patient-grouped 5-fold CV, per DS's deliverable)
- **Verification:** both artifacts independently verified by ML Engineering
  before integration — see `reports/week6_model_verification.md` for full
  evidence (reload-identity, schema match, reproduced metrics, confirmed bug).

### Training Data
- Same source dataset. Split: `GroupShuffleSplit(test_size=0.2, random_state=42)`,
  grouped by `patient_id` — **0 verified patient overlap** between train/test.
- Train / test size: 3,771 / 966.

### Features
11 features — 6 numerical, 5 categorical. Deliberately smaller than the
baseline's 16 — Data Science confirmed via ablation (correlation test,
bootstrap CI) that the baseline's additional engineered features
(`lead_time_bucket`, `distance_bucket`, `is_first_time_patient`,
`is_weekend_appointment`) were not part of their final feature set. See
`configs/config.yaml` under `candidate_models:`.

`historical_no_show_rate` is derived upstream (`previous_no_shows /
previous_appointments`, 0.0 fallback), not a raw input field.

### Evaluation (held-out test set, n=966, independently reproduced — Week 6 single-seed result)

| Metric | Logistic Regression | Gradient Boosting |
|---|---:|---:|
| Accuracy | 63.46% | 65.11% |
| ROC-AUC | 0.6825 | 0.6941 |
| Recall (No-Show) | 66.0% | 66.7% |
| Confusion matrix | TN294/FP189/FN164/TP319 | TN307/FP176/FN161/TP322 |

**⚠️ Superseded by Week 7 multi-seed testing — see below.** The table
above reflects only `random_state=42`. Do not read it as "GB is better."

### Week 7 update: multi-seed stability test retracts the Week 6 "GB is better" conclusion — pending methodology confirmation

Data Science reported re-running the AUC comparison across 5 random seeds:

| Seed | DS-reported AUC gap (GB − LR) |
|---|---:|
| 42 | +0.0116 |
| 7 | +0.0015 |
| 123 | −0.0028 |
| 2024 | −0.0127 |
| 99 | −0.0077 |

**DS-reported mean: −0.002.** GB wins on 2 of 5 seeds and loses on 3 —
statistically indistinguishable from a coin flip, per their report.

**⚠️ Independently re-tested — result does not match, for a specific,
identified reason.** Running the same 5 seeds against the actual
`.joblib` artifacts in this repo (same `GroupShuffleSplit` methodology)
produces **GB winning 5/5, mean gap +0.0121** — matching Data Science's
number only at seed 42. Investigation found the GB artifact is fully
deterministic (`subsample=1.0`, `max_features=None`, fixed
`random_state=42`) — so evaluating the *same fitted model* across
different test splits cannot reproduce sign-flipping results. This
strongly suggests Data Science's test **retrained both models per seed**
(a test of training-process stability), while my test evaluated the
*existing, already-fitted* artifacts across different test partitions (a
test of test-set-sampling robustness) — two different, both legitimate,
questions with two different answers. See
`reports/week7_multiseed_verification.md` for the full investigation.

**Practical takeaway pending Data Science's confirmation of their
methodology:** the retraction may still be correct — if a future GB
refit is genuinely as likely to underperform LR as outperform it, that's
important and this pipeline should keep treating both models as
interchangeable, which it already does. But it does *not* mean the two
specific artifacts currently in `models/candidates/` are unstable
against each other — on this pipeline's actual deployed pair, GB
consistently wins across test-set variation. Both facts can be true at
once; neither is being asserted as the final word here.

**Either way, this pipeline requires no code change.** `predict()`'s
dispatcher was already built in Week 6 to route by caller-selected
`model_name` with no hardcoded preference — see `src/models/predict.py`.

**Feature importance (GB, seed 42 only):** `booking_lead_days` = 66.5% of
total importance. Given the multi-seed instability above, this
single-feature dominance is now read as a likely contributor to GB's
seed-sensitivity, not just a general caution.

### Week 7 update: segment-level weak spots identified

Regardless of which model is used, both show reduced reliability on:

| Segment | AUC | Recall (No-Show) | vs. overall |
|---|---:|---:|---|
| Age 65+ | 0.664 | 0.595 | below overall (0.694 / 0.667) |
| Specialist Consultation | 0.616 | 0.553 | weakest segment found |

Any confidence-flagging logic added to this pipeline should treat
predictions in these two segments as lower-trust, regardless of which
model produced them.

---

## Intended Use (all models)

Decision-support tool for HealthConnect staff to flag appointments at
elevated risk of a no-show, so reminder/intervention effort can be
prioritized. **Not** intended as an automatic system for denying or
altering care, and not validated on real patient data.

## Confirmed Integration Bug (candidates only, fixed)

Raw `NaN` in `reminder_channel` is silently treated by the fitted
`OneHotEncoder` as an unseen category rather than the fitted `"None"`
category, producing a materially different, silently-degraded prediction
on 27.3% of the raw dataset. Fixed via a mandatory preparation gate
(`src/models/candidate_models.py::prepare_candidate_input`) and pinned
with a regression test. Full details: `reports/week6_model_verification.md`.

`waiting_time_minutes` is explicitly excluded from all three models
(confirmed data leakage — populated even for appointments the patient
never attended).

## Limitations

**Baseline-specific:**
- Train/test split is plain stratified, **not** group-aware — 1,394 of
  1,696 patients appear in more than one row, so the same patient can
  appear in both folds. Not yet fixed (Week 7 item).
- Not directly comparable to the candidates' metrics, since the split
  methodology differs.

**Candidate-specific (updated per Week 7 Data Science handoff):**
- **No model recommendation currently stands.** The Week 6 "GB is better"
  conclusion was a single-seed artifact — see the multi-seed table above.
  Data Science is redoing the recommendation with multi-seed evidence in
  hand; treat neither model as the pipeline's default until that lands.
- Two segments (age 65+, Specialist Consultation) show materially reduced
  reliability for both models — see Week 7 update above.
- Uncertain zone (predictions in 0.4–0.6) was 37.1% for LR / 45.7% for GB
  at seed 42 — large regardless of model, a likely feature-set ceiling
  per Data Science's own interpretation.
- No calibration check performed yet.

**Shared across all models:**
- Trained on **synthetic/anonymized** data — performance on real
  HealthConnect patients is unverified.
- Two feature groups (reminder timing, historical-count construction) are
  included pending confirmation of modeling assumptions not stated in the
  source data dictionary — **still unresolved as of Week 7; being
  escalated to Project Management this week per Data Science.**
- No causal claims — all relationships are correlational.

## Ethical Considerations

- Model output is a probability + risk category, framed as decision
  support, not an automated action.
- No personally identifying information is used as a feature.
- Performance across demographic groups (e.g. `gender`, `age`) has not
  yet been separately evaluated for any of the three models.
