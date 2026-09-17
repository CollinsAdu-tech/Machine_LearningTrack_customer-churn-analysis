# Week 6 Project Summary — Machine Learning Engineering Track

**1. What I planned to accomplish:**
Integrate the Data Science track's improved candidate model(s) into the
existing Week 5 ML pipeline and validate that the integrated system works
correctly and reproducibly — not rebuild the Week 5 pipeline.

**2. What I completed:**
Received, independently verified, and integrated two Data Science model
artifacts (Logistic Regression baseline, tuned Gradient Boosting
candidate) into the pipeline via a new `candidate_models.py` module and an
extended `predict()` dispatcher. Found and fixed a real integration bug.
Built a fresh-environment reproducibility check with pinned dependencies.
Documented all of it.

**3. What I improved from Week 5:**
The Week 5 pipeline could only produce predictions from one hardcoded
baseline model. It can now route between three models (baseline, LR
candidate, GB candidate) through the same validated interface, with
`requirements.txt` now pinned to exact working versions (previously
unpinned).

**4. What I integrated:**
Two `.joblib` model artifacts and their required preprocessing
assumptions, reconciled against a distinct 11-feature schema that
partially overlaps with, but is smaller than, the Week 5 baseline's
16-feature schema.

**5. Which track(s) I collaborated with:**
Data Science.

**6. What was exchanged:**
Received: model artifacts, exact schema, fitted category order, split
methodology, full evaluation results, feature importance, limitations.
Provided: an integrated model interface, a discovered and reported
integration bug with reproduction steps, and independent confirmation
that their reported metrics are reproducible from raw data.

**7. What changed as a result:**
A silent prediction-quality bug affecting 27.3% of the dataset (raw
`reminder_channel` nulls mishandled by the shared preprocessor) was found
before it could reach any downstream use, and fixed with a permanent,
automated regression test. Data Science received a specific, actionable
finding for their own artifact documentation.

**8. Key findings or development outcomes:**
- Every number in Data Science's Week 6 deliverable was independently
  reproduced exactly from the raw dataset — the handoff was fully
  trustworthy
- The one gap found (`reminder_channel` null handling) was in an
  implicit assumption, not a misrepresented result
- The pipeline is now verified reproducible from a genuinely clean
  environment, not just re-tested in the existing one

**9. Major challenges:**
Reconciling two different feature schemas (Week 5 baseline's 16 features
vs. the candidate models' 11) without collapsing them into one — they
represent genuinely different, deliberate modeling decisions and forcing
a single shared schema would have misrepresented what each model actually
expects.

**10. Important decisions:**
Kept the Week 5 baseline and the Week 6 candidate models as parallel,
independently-testable paths rather than replacing the baseline outright
— the baseline's own split methodology (plain stratified, not
group-aware) still needs a fix before it's directly comparable to the
candidates, which is flagged as Week 7 work rather than rushed now.

**11. Remaining issues:**
Dispatcher-level model selection (which model becomes "the" production
choice) is intentionally not decided here — pending Project
Management/business input, per Data Science's own caution against
treating the GB improvement as settled. The Week 5 baseline's split
method still needs upgrading to `GroupShuffleSplit` for a fair three-way
comparison.

**12. Contribution to the overall HealthConnect project:**
The system can now serve validated no-show predictions using either
track's model output, tested and documented well enough for another
engineer (or Week 7 testing) to pick up without re-deriving what was
already checked.

**13. Proposed focus for Week 7:**
See `reports/week7_ml_engineering_testing_plan.md` — group-aware split for
the baseline, schema-drift detection, runtime benchmarking, a multi-model
comparison utility, and following up on the two still-open conditional
feature assumptions.
