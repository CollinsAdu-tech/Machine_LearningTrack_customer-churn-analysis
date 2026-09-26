# HealthConnect No-Show Prediction — Model Card

**⭐ FINAL MODEL (Week 8): Gradient Boosting** — `model_name="gradient_boosting"`.
Two other models remain available and fully functional
(`"baseline"`, `"logistic_regression"`) for comparison and reproducibility,
but Gradient Boosting is the confirmed final selection for the
HealthConnect no-show prediction system. See "Final Model Decision" below
for the full reasoning and honesty notes before treating this as a simple
"GB is better" claim.

---

## Final Model: Gradient Boosting (Week 6 artifact, finalized Week 8)

- **Model type:** Gradient Boosting Classifier (scikit-learn)
- **model_name:** `"gradient_boosting"`
- **Hyperparameters:** `n_estimators=100, max_depth=2, learning_rate=0.05`
  (selected via `GridSearchCV` with patient-grouped 5-fold CV, per Data
  Science's Week 6 deliverable)
- **Artifact:** `models/candidates/healthconnect_gb_pipeline.joblib` —
  unchanged since Week 6; no new file was delivered for Week 8, none was
  needed
- **Verification:** independently verified before integration (Week 6:
  reload-identity, schema match, reproduced metrics, confirmed/fixed a
  null-handling bug — `reports/week6_model_verification.md`) and the
  Week 7/8 multi-seed question independently confirmed rather than
  taken on trust — see "Final Model Decision" below

### Training Data
- `HealthConnect_Appointment_Data.csv` — 5,000 fictional, anonymized
  appointment records, filtered to `{Attended, No-Show}` (Cancelled
  excluded, ~5.3%).
- Split: `GroupShuffleSplit(test_size=0.2, random_state=42)`, grouped by
  `patient_id` — **0 verified patient overlap** between train/test.
- Train / test size: 3,771 / 966.

### Features
11 features — 6 numerical, 5 categorical:

**Numerical:** `age`, `booking_lead_days`, `previous_appointments`,
`previous_no_shows`, `historical_no_show_rate` (derived upstream:
`previous_no_shows / previous_appointments`, 0.0 fallback),
`distance_to_clinic_km`.

**Categorical:** `gender`, `appointment_type`, `appointment_day`,
`appointment_time`, `reminder_channel`.

Deliberately smaller than the Week 5 baseline's 16 features — Data
Science confirmed via ablation that the baseline's additional engineered
features were not part of the final feature set. **Confirmed unchanged
since Week 6** — no schema updates needed for Week 8.

### Final Evaluation Metrics (official test set, n=966)

| Metric | Value |
|---|---:|
| Accuracy | 65.11% |
| ROC-AUC | 0.6941 |
| Recall (No-Show) | 66.7% |
| Confusion matrix | TN 307 / FP 176 / FN 161 / TP 322 |

**Honesty note (stated explicitly by Data Science, carried through
here):** across 5 retrained-per-seed comparisons, GB's AUC ranged
0.648–0.704 (mean ≈0.677). **0.6941 is the number for this specific
trained artifact — not a guaranteed floor for any future retrain.** If
this model is ever retrained on different data, expect ±0.02–0.03
variability in ROC-AUC.

---

## Final Model Decision — the multi-seed question, resolved

This is worth documenting precisely, because the path to this decision
involved a genuine disagreement between two correct-but-different tests,
not a simple confirmation.

**What happened:** Data Science's Week 7 multi-seed test reported GB
winning only 2 of 5 seeds (mean AUC gap −0.002) — appearing to retract
the Week 6 "GB is better" finding. ML Engineering independently re-ran
the same 5 seeds against the actual delivered artifacts and got a
different result: **GB winning 5/5 (mean gap +0.0121).**

**Root cause, confirmed by both sides:** Data Science's test **retrained
both models from scratch on each seed's training split** (measuring
training-process stability). ML Engineering's re-test evaluated the
**same fixed, already-fitted artifacts** across different test
partitions (measuring test-set-sampling robustness). These are different
questions, and both results are correct for the question each one asked.

**Independently confirmed, not just accepted:** Data Science demonstrated
this by resampling the fixed test set 5 times without retraining,
reporting GB winning 5/5. ML Engineering reproduced this exact experiment
independently against the same `.joblib` files:

```
Resample 1: AUC_LR=0.6621  AUC_GB=0.6899  GB wins
Resample 2: AUC_LR=0.6865  AUC_GB=0.6982  GB wins
Resample 3: AUC_LR=0.6926  AUC_GB=0.6958  GB wins
Resample 4: AUC_LR=0.6914  AUC_GB=0.6968  GB wins
Resample 5: AUC_LR=0.6735  AUC_GB=0.6877  GB wins
```

GB wins 5/5, values in the same range Data Science reported. This
confirms the explanation rather than resolving it by assertion.

**The defensible final claim, stated at the correct scope:** *not*
"Gradient Boosting is a fundamentally better algorithm for this
problem" — that claim did not survive the retrain-stability test. The
defensible claim is: **"on the actual available data, this specific
trained GB model outperforms this specific trained LR model,
consistently."** That narrower claim is true, independently confirmed,
and sufficient to finalize the model selection. Full writeup:
`reports/week8_multiseed_resolution.md`, `reports/week8_final_model_decision.md`.

**Feature importance (GB):** `booking_lead_days` = 66.5% of total
importance. Given the retrain-instability finding, this single-feature
dominance is read as a likely contributor to why GB's advantage doesn't
survive being retrained on a different data slice — worth investigating
if this model is ever revisited.

---

## What This Model Can and Cannot Be Used For

Recorded near-verbatim from Data Science's final Week 8 statement, not
paraphrased, so this pipeline's documentation says exactly what they
intended:

> This model estimates the probability that a scheduled HealthConnect
> appointment will result in a no-show, using patient demographics,
> appointment characteristics, booking behavior, and historical
> attendance patterns available at scheduling time. It is suitable for
> **flagging appointments at elevated risk for optional preventive
> follow-up** (e.g., an additional reminder). It is **not suitable** for
> automated decisions with direct consequences for patients (e.g.,
> automatic cancellation, denial of service, or overbooking without
> human review), for use as a diagnostic or clinical tool, or for
> generalizing to populations outside this fictional dataset.
> Approximately 37–46% of predictions fall in an uncertain probability
> range (0.4–0.6) where the model performs close to chance; predictions
> in this range should be treated as low-confidence. Performance is
> measurably weaker for patients aged 65+ (AUC 0.664, Recall 0.595) and
> Specialist Consultation appointments (AUC 0.616, Recall 0.553) —
> outputs for these segments should carry a lower-confidence flag if
> your pipeline supports it. This model has not been evaluated for
> fairness or bias across demographic groups beyond the segment check
> reported, has not been calibration-tested, and should not be
> considered production-ready without further validation on real
> (non-synthetic) data.

---

## Other Available Models (kept for comparison, not the final selection)

### Baseline (Week 5)
- **Model type:** Logistic Regression, `model_name="baseline"`
- **Hyperparameters:** `C=1.0`, `class_weight=None`, `max_iter=1000`, `random_state=42`
- **Artifact:** `models/model_v1.joblib`; trained by this repo (`src/models/train.py`)
- Split: 80/20 **plain stratified** (not group-aware — a known
  limitation, see below), 3,789 train / 948 test
- 16 features (6 numerical, 7 categorical, 3 boolean)
- Evaluation: Recall (No-Show) 0.676, ROC-AUC 0.667, Accuracy 0.615,
  confusion matrix TN=255/FP=208/FN=157/TP=328

### Logistic Regression candidate (Week 6)
- `model_name="logistic_regression"`, `models/candidates/healthconnect_logreg_pipeline.joblib`
- Same 11-feature schema and split as the final GB model
- Evaluation: Accuracy 63.46%, ROC-AUC 0.6825, Recall (No-Show) 66.0%,
  confusion matrix TN294/FP189/FN164/TP319
- Kept available for comparison and reproducibility, not the final choice

---

## Confirmed Integration Bug (GB and LR candidates, fixed)

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

**Final model (Gradient Boosting):**
- Reliably beats LR on the actual available data (independently
  confirmed), but this advantage is **not guaranteed to survive a
  retrain** on different data — see Final Model Decision above.
- Two segments (age 65+, Specialist Consultation) show materially reduced
  reliability.
- Uncertain zone (predictions in 0.4–0.6): 45.7% of predictions at seed 42.
- Single feature (`booking_lead_days`) drives 66.5% of importance —
  plausibly connected to the model's retrain-instability.
- No calibration check performed.

**Baseline-specific:**
- Train/test split is plain stratified, not group-aware — 1,394 of 1,696
  patients appear in more than one row, so the same patient can appear in
  both folds. Not fixed as of Week 8 — not required for the final model
  since it's not the baseline that was selected.
- Not directly comparable to the candidates' metrics (different split
  methodology).

**Shared across all models:**
- Trained on **synthetic/anonymized** data — performance on real
  HealthConnect patients is unverified.
- Two previously-open assumptions (reminder timing, historical-count
  construction) are **closed as of Week 8, but as documented assumptions,
  not independently confirmed facts** — there is no live business/PM
  stakeholder in this exercise to formally verify them:
  - **Reminder timing:** assumed prediction occurs 24–48 hours before the
    appointment (after reminders are typically sent).
  - **Historical-count construction:** assumed `previous_appointments` /
    `previous_no_shows` are computed strictly prior to and excluding the
    current appointment.
- No causal claims — all relationships are correlational.

## Ethical Considerations

- Model output is a probability + risk category, framed as decision
  support, not an automated action.
- No personally identifying information is used as a feature.
- Performance across demographic groups (e.g. `gender`, `age`) has not
  been evaluated for fairness/bias beyond the two segment findings
  reported above (age 65+, Specialist Consultation) — explicitly flagged
  by Data Science as not yet fairness-tested.
- Not production-ready without further validation on real (non-synthetic)
  data, per Data Science's own final statement.
