# Week 7 — Mandatory HC-POD Cross-Track Testing & Refinement Record

Per the assignment's Section 9 template. This is a genuine Test → Finding
→ Action → Retest cycle, not a discussion or status update.

**1. Track collaborated with:**
Data Science

**2. Project dependency:**
ML Engineering needed to confirm whether the Week 7 model-selection
retraction (GB no longer recommended over LR) was itself reliable before
updating the pipeline's documentation and guidance to reflect it.

**3. Component/output being tested:**
Data Science's Week 7 multi-seed AUC stability claim (GB wins 2/5 seeds,
mean gap −0.002) — tested against the two `.joblib` artifacts already
integrated into this pipeline.

**4. Information/output received:**
The 5-seed AUC gap table, the retraction of the Week 6 "GB is better"
recommendation, and two new segment-level weak-spot findings (age 65+,
Specialist Consultation).

**5. Information/output provided:**
An independent re-run of the exact same 5-seed test against the actual
delivered artifacts, plus a root-cause investigation (checking the GB
model's fitted hyperparameters for remaining stochasticity) explaining
*why* the two result sets likely diverge, plus a specific, answerable
clarifying question sent back rather than a vague "numbers don't match."

**6. Testing activity completed:**
Reproduced Data Science's `GroupShuffleSplit(test_size=0.2,
random_state=seed)` methodology across the same 5 seeds
(42, 7, 123, 2024, 99), using the actual `healthconnect_logreg_pipeline.joblib`
and `healthconnect_gb_pipeline.joblib` files already in
`models/candidates/`. Additionally inspected the fitted GB classifier's
`subsample`, `max_features`, and `random_state` parameters to determine
whether the artifact itself retains any seed-dependent randomness.

**7. Issue or finding identified:**
**The results did not match.** Data Science reported GB winning 2/5
seeds (mean gap −0.002); this pipeline's independent re-test found GB
winning 5/5 seeds (mean gap +0.0121) against the same artifacts. Root
cause identified: the GB artifact is fully deterministic
(`subsample=1.0`, `max_features=None`, fixed `random_state=42`), meaning
this test could only vary which patients landed in the test set — the
model itself never changed across the 5 runs. This strongly suggests
Data Science's test retrained both models per seed (a different, and
arguably more decision-relevant, experiment than evaluating fixed
artifacts across test-set resamples).

**8. Refinement/action taken:**
No pipeline code was changed, since `predict()`'s dispatcher already
treats both models as fully interchangeable regardless of which
conclusion is correct. Documentation (`docs/model_card.md`,
`reports/week7_ds_handoff_update.md`) was updated to present both result
sets transparently, with neither asserted as final, plus a specific
clarifying question sent to Data Science
(`reports/week7_ds_feedback.md`) asking whether their test retrained
per-seed or reused fixed artifacts.

**9. Retest result:**
Pending Data Science's answer to the methodology question. This
pipeline's own re-test is fully reproducible and documented in
`reports/week7_multiseed_verification.md` — re-running it against the
same artifact files will always produce the same 5/5, +0.0121 result,
since the artifacts are deterministic.

**10. What changed as a result:**
The pipeline's documentation no longer treats Data Science's retraction
as an unverified fact copied into the model card — it's now presented
alongside the independent re-test and the specific reason the two
diverge, so anyone reading the model card understands there's an open
methodological question rather than a settled number. This also
surfaced a genuinely useful, separate fact: the two specific artifacts
currently deployed in this pipeline are stable relative to each other
across test-set sampling, regardless of how the broader training-process
stability question resolves.

**11. Evidence:**
- Code/execution: independent 5-seed reproduction (results in
  `reports/week7_multiseed_verification.md`)
- Documentation: `docs/model_card.md` (updated), `reports/week7_ds_handoff_update.md`,
  `reports/week7_multiseed_verification.md`, `reports/week7_ds_feedback.md`
- Artifacts tested: `models/candidates/healthconnect_logreg_pipeline.joblib`,
  `models/candidates/healthconnect_gb_pipeline.joblib` (unchanged, verified
  deterministic)

**12. How the activity improved the overall HealthConnect solution:**
Prevented a documentation update based on an unverified claim from
propagating into the model card and downstream decision-making. Applied
the same verify-before-trust standard used throughout this project
(Weeks 5–6) to Data Science's Week 7 output specifically, catching a
real discrepancy before it could lead to an incorrect final model
recommendation being locked in without scrutiny.
