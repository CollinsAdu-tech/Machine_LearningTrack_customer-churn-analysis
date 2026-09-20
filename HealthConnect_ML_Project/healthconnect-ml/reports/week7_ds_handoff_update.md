# Week 7 — Data Science Handoff Update

## Purpose

Records what changed in the Week 7 Data Science handoff relative to Week
6, and what it does/doesn't require changing in this pipeline. Per the
same standard as Week 6: claims are checked against what's actually in
this repo before being accepted, not assumed correct because they came
from a prior collaborator.

## What changed

| Item | Week 6 handoff | Week 7 update |
|---|---|---|
| Model recommendation | GB recommended as candidate, with caution | **Reported as retracted.** Multi-seed test shows mean AUC gap (GB−LR) = −0.002 across 5 seeds; GB wins 2, loses 3. **Independently re-tested by ML Engineering — did not reproduce.** See below. |
| Statistical significance | Bootstrap 95% CI [+0.0003, +0.0229] (single split, seed 42) | Superseded — that CI didn't hold once the split itself was varied |
| Segment performance | Not assessed | **New finding:** age 65+ and Specialist Consultation segments show meaningfully lower AUC/Recall for both models |
| Input schema, one-hot categories, output contract | As verified in Week 6 | **Unchanged** — confirmed by Data Science, no re-verification needed on our end since the artifacts themselves haven't changed |
| Two conditional-feature assumptions (reminder timing, historical-count construction) | Open | **Still open** — Data Science says these are being escalated to PM this week, not resolved yet |
| `reminder_channel` bug feedback | Sent (`week6_ds_feedback.md`) | Data Science reports not having received/retained it — resent this cycle |
| Re-fit notification agreement | Requested | Confirmed — Data Science will notify before either `.joblib` changes |
| Final candidate-model recommendation | Delivered (GB) | **Explicitly retracted, not yet redone** — Data Science says this needs to be rebuilt with multi-seed evidence in hand |

## What this means for this pipeline — checked, not assumed

**Critical finding, not just accepted: the multi-seed retraction was
independently re-tested and did not reproduce as reported.** Running the
same 5 seeds against the actual `.joblib` artifacts in this repo produces
GB winning 5/5 (mean +0.0121), not 2/5 (mean −0.002) as reported.
Investigation found the GB artifact is fully deterministic
(`subsample=1.0`, fixed `random_state=42`) — meaning Data Science's test
almost certainly retrained both models per seed, while this
verification evaluated the same fixed, already-delivered artifacts across
different test splits. These are two different, both legitimate,
questions — see `reports/week7_multiseed_verification.md` for the full
investigation and root-cause analysis. **This has been sent back to Data
Science as a clarifying question, not silently accepted or silently
contradicted.**

**No code change required for the model-selection retraction, regardless
of which interpretation turns out correct.**
`src/models/predict.py`'s dispatcher was already built in Week 6 to treat
`model_name` as a pure caller choice — `"baseline"`, `"logistic_regression"`,
or `"gradient_boosting"` are handled identically in the routing logic,
with no hardcoded preference for GB anywhere in `predict()`,
`candidate_models.py`, or `configs/config.yaml`. Verified by search:

```
grep -rn "gradient_boosting is the\|GB is the\|recommend" src/ configs/
```
returns nothing suggesting a hardcoded preference. This was good
foundational design, not luck — the dispatcher's whole point was
supporting both models "interchangeably," which is exactly what this
update now requires explicitly.

**Documentation change required, and made:** `docs/model_card.md` and
`README.md` both described GB using DS's Week 6 language ("recommended
candidate," a specific significant AUC gap). That language has been
corrected to reflect the retraction — see the diffs in this commit.

**No test changes required.** All 21 tests in
`tests/test_candidate_models.py` test both models' mechanical behavior
(schema, reload-identity, the null-handling bug fix, reproduced Week 6
metrics) — none of them assert or depend on GB being "better." The
reproduced-metrics test
(`test_reported_evaluation_metrics_are_reproducible`) still correctly
reproduces the Week 6 seed-42 numbers, which remain factually accurate
for that specific seed — only the *conclusion* drawn from them was
retracted, not the numbers themselves. No update needed there.

## Segment weak spots — not yet built into the pipeline

Data Science's new finding (age 65+, Specialist Consultation both show
reduced Recall/AUC) is not currently surfaced anywhere in `predict()`'s
output. The output contract (`prediction`, `label`,
`no_show_probability`, `risk_category`, `model_used`) has no field
indicating segment-based confidence. This is noted as a candidate Week 7
enhancement (see `reports/week7_ml_engineering_testing_plan.md`), not yet
implemented — flagging the gap rather than quietly leaving it
undocumented.

## Outstanding action taken this cycle

- Resent `reports/week6_ds_feedback.md` to Data Science (they reported
  not having it).
- Confirmed via code search that no pipeline logic needs to change for
  the retraction — only documentation.
- Updated `docs/model_card.md` and `README.md` to remove "GB is
  recommended" framing and replace with the multi-seed table and
  retraction notice.
- **Independently re-ran the multi-seed test against the actual
  delivered artifacts** — did not reproduce Data Science's reported
  numbers (GB won 5/5 here vs. 2/5 reported). Identified the likely
  methodological cause (retrain-per-seed vs. fixed-artifact-evaluated-
  on-different-splits) and sent a specific clarifying question back
  rather than accepting or silently overriding either result.

## Still waiting on Data Science

- The redone final candidate-model recommendation, once multi-seed
  evidence is fully incorporated.
- Resolution (not just escalation status) on the two open conditional-
  feature assumptions.
- **Confirmation of the multi-seed test methodology** — did it retrain
  both models per seed, or evaluate the same fixed artifacts across
  different test splits? This determines which of the two conflicting
  result sets (theirs or this repo's independent re-test) is answering
  the question that actually matters for a final recommendation.
