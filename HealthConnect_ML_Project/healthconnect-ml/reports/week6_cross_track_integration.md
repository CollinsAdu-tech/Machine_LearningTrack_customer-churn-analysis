# Week 6 — Mandatory Cross-Track Integration Record

Per the assignment's Section 9 template.

**1. Track collaborated with:**
Data Science

**2. Project dependency:**
ML Engineering needed the finalized candidate model artifact(s) and
refined feature specification before the Week 5 pipeline could move from
"baseline established" to "integrated" — the explicit Week 6 objective.

**3. Information/output received:**
- Two verified `sklearn.Pipeline` artifacts: `healthconnect_logreg_pipeline.joblib`
  (baseline) and `healthconnect_gb_pipeline.joblib` (tuned candidate)
- Exact input schema (6 numeric + 5 categorical features) and fitted
  one-hot category order
- Confirmed train/test split methodology (`GroupShuffleSplit`, grouped by
  `patient_id`, 0 patient overlap)
- Full evaluation results for both models (accuracy, ROC-AUC, confusion
  matrices, bootstrap significance testing)
- Feature importance ranking, error analysis, and an explicit list of
  limitations and Week 7 testing requirements

**4. Information/output provided:**
- An integrated model interface (`src/models/candidate_models.py`,
  `predict_with_candidate()`) making both models callable through a
  single, validated entry point
- A discovered integration bug (raw `reminder_channel` nulls silently
  mishandled by the shared `OneHotEncoder`) — reported back via
  `reports/week6_ds_feedback.md`, with reproduction steps, so it's fixed
  for any consumer of these artifacts, not just this pipeline
- Independent reproduction of all of Data Science's reported evaluation
  metrics, confirming their numbers rather than just trusting them

**5. Integration activity completed:**
Both candidate models were loaded, verified against their claimed
behavior (schema, reload-identity, reported metrics), and wired into the
existing Week 5 `predict()` interface as a routable dispatcher
(`model_name="baseline" | "logistic_regression" | "gradient_boosting"`).
A mandatory input-preparation step was built to bridge a real
compatibility gap found between the artifacts' stated behavior and their
actual behavior on raw data.

**6. What changed as a result:**
- The HealthConnect pipeline can now produce a validated prediction using
  either of Data Science's models, not just the original Week 5 baseline
- A silent, 27.3%-of-dataset-affecting prediction-quality bug was found
  and fixed before it could reach any downstream use
- Data Science now has documented evidence their reported numbers are
  independently reproducible, and a specific, actionable fix to consider
  for their own pipeline documentation

**7. Evidence:**
- Code: `src/models/candidate_models.py`, `src/models/predict.py`
  (dispatcher)
- Tests: `tests/test_candidate_models.py` (21 tests),
  `tests/test_predict_dispatcher.py` (13 tests) — full suite 67/67 passing
- Documentation: `reports/week6_model_verification.md`,
  `reports/week6_ds_feedback.md`, `reports/week6_issue_log.md`
- Model artifacts: `models/candidates/*.joblib`
