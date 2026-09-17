# Week 7 Testing Plan — Machine Learning Engineering Track

Distinct from Data Science's own Week 7 testing requirements (multi-seed
stability, calibration, threshold sensitivity — see their deliverable
Section 11). This plan covers what's specifically an ML Engineering
responsibility: pipeline behavior, not model behavior.

## 1. Cross-environment reproducibility (extends Week 6's initial check)

Week 6 verified reproducibility on the same OS/Python version as
development. Week 7 should extend this to:
- A different Python minor version (e.g., 3.11 vs. the 3.12 used in Week 6)
- Confirm the pinned `requirements.txt` versions still resolve and install
  cleanly

## 2. Group-aware split in the Week 5 baseline (carried over from Issue Log #3)

The Week 5 baseline (`src/models/train.py`) still uses a plain stratified
split, not the `GroupShuffleSplit` DS's candidate models use. This means
the baseline's own reported metrics (Accuracy 0.615, ROC-AUC 0.667) are
not directly comparable to the candidate models' metrics (0.6346/0.6511
accuracy) — they were evaluated under different splitting methodology.
Week 7 should implement `GroupShuffleSplit` in the baseline's `split_data()`
and re-evaluate, so all three models (baseline, LR candidate, GB
candidate) are compared on identical footing.

## 3. Schema-drift detection

Currently, `candidate_models.py` and `predict.py` will raise a clear error
if a required field is missing, but there's no automated check for the
reverse case: if the underlying `.joblib` artifacts are ever regenerated
by Data Science with a different feature set or different fitted
categories, nothing currently detects that automatically — it would only
surface as a downstream prediction error or, worse, a silently different
result. Week 7 should add a startup-time schema check: on loading a
candidate model, verify its `ColumnTransformer`'s expected columns and
fitted categories match what `config.yaml` declares, and fail loudly if
they don't.

## 4. Runtime / performance baseline

No latency or throughput measurement has been taken for either candidate
model. Week 7 should establish a basic benchmark (single-prediction
latency, batch-prediction throughput for both models) — not because
there's a known performance problem, but because "integrated and
validated" should include knowing the pipeline's basic operational
characteristics before any broader testing claims it's ready.

## 5. Multi-model comparison harness

Right now, comparing baseline vs. LR vs. GB requires manually calling
`predict()` three times with different `model_name` values. Week 7 could
add a small utility that runs the same batch of records through all three
models and returns a side-by-side comparison — useful for Data Science's
own multi-seed stability testing (item 1 in their Week 7 list) and for
Project Management's eventual model-selection decision.

## 6. Resolve the two open conditional-feature assumptions

Not an ML Engineering task to resolve alone (needs Project Management /
business input, per Data Science's own handoff), but Week 7 should
include following up on whether these have been settled:
- Whether `reminder_sent`/`reminder_channel` are legitimately available at
  the intended prediction time
- Whether `previous_appointments`/`previous_no_shows` exclude the current
  appointment

If resolved unfavorably, `config.yaml`'s `conditional_features` section
(Week 5 baseline) and the equivalent assumption in the candidate models'
feature set would both need revisiting.

## 7. End-to-end integration test with a downstream consumer

All current tests call `predict()` directly in-process. Week 7 should
add at least one test that simulates how a real downstream
system would call this (e.g., a simple script that takes a CSV of new
appointment records, runs each through `predict()`, and writes results) —
closer to how this would actually be used operationally than a unit test
calling the function directly.

## Explicitly out of scope for Week 7 (per Data Science's own caution)

Treating GB as the final production model. Their Week 6 deliverable is
explicit that the improvement is small, single-seed, and doesn't shrink
the uncertain zone — that decision waits on their Week 7 stability
testing and a Project Management call, not something ML Engineering
testing should presuppose.
