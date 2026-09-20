# Week 7 — Independent Verification of Data Science's Multi-Seed Retraction

## Why this exists

Data Science's Week 7 update retracted the Week 6 "GB beats LR"
conclusion based on a multi-seed test (GB wins 2/5 seeds, mean AUC gap
−0.002). Per this project's own established standard — every Week 6
claim was independently reproduced before being trusted, not just
accepted — the same bar applies here. **Updating documentation based on
a report, without re-running the test, would not meet that standard.**
This document records what was actually run.

## What was tested

Using the two `.joblib` artifacts already in this repo
(`models/candidates/healthconnect_logreg_pipeline.joblib`,
`healthconnect_gb_pipeline.joblib`), evaluated on 5 different
`GroupShuffleSplit(test_size=0.2, random_state=seed)` test partitions,
using the same seeds Data Science reported:

```python
seeds = [42, 7, 123, 2024, 99]
# for each seed: split, then pipeline.predict_proba(X_test) -> roc_auc_score
```

## Result — this does NOT match Data Science's reported table

| Seed | My LR AUC | My GB AUC | My gap (GB−LR) | DS-reported gap |
|---|---:|---:|---:|---:|
| 42 | 0.6825 | 0.6941 | **+0.0116** | +0.0116 ✅ matches |
| 7 | 0.7111 | 0.7264 | **+0.0152** | +0.0015 ❌ differs |
| 123 | 0.6736 | 0.6905 | **+0.0169** | −0.0028 ❌ sign differs |
| 2024 | 0.6701 | 0.6778 | **+0.0078** | −0.0127 ❌ sign differs |
| 99 | 0.6888 | 0.6978 | **+0.0090** | −0.0077 ❌ sign differs |

**My mean gap: +0.0121. GB wins 5/5 seeds.**
**DS-reported mean gap: −0.002. GB wins 2/5 seeds.**

Only seed 42 matches — which is the exact seed both this repo's Week 6
verification and Data Science's original Week 6 deliverable both used.
The other four diverge completely, including in *sign*.

## Root cause investigation — before reporting this back, not after

Checked whether the GB artifact itself has any remaining randomness that
could explain seed-to-seed variation on its own:

```python
model.random_state   # 42
model.subsample       # 1.0  <- no stochastic row subsampling per tree
model.max_features    # None <- no random feature subsampling per split
```

**Finding: this specific fitted GB pipeline is fully deterministic.**
`subsample=1.0` means no randomness in which rows each boosting stage
sees; `max_features=None` means no randomness in which features each
split considers. Given identical input `X`, `predict_proba()` will always
return identical output — there is no seed-dependence left in the
*already-fitted* artifact itself.

**This means my test could only ever vary one thing: which patients
ended up in the test set.** The model itself never changed across my 5
runs — I evaluated the same fixed, already-fitted pipeline against 5
different test slices.

**For Data Science to get results that flip sign across seeds, the model
itself must have been different in each of their 5 runs** — i.e., they
almost certainly **retrained both LogisticRegression and
GradientBoostingClassifier from scratch on each seed's own training
split**, not reused the fixed artifacts I have. A freshly retrained GB on
a different training subset would have different tree structures
entirely (different split points, different feature interactions
learned), which is a much larger source of variation than which rows
happen to land in the test set.

## What this actually means — two different, both legitimate questions

| | My test | Data Science's likely test |
|---|---|---|
| **Question answered** | "Given these exact two delivered artifacts, how robust is GB's advantage to which patients are in the test set?" | "If the whole training pipeline is rerun with a different split, is GB reliably better than LR?" |
| **Answer** | Robust — GB wins 5/5 on this fixed pair, mean +0.012 | Not robust — roughly a coin flip, mean −0.002 |
| **What it tells you** | These two specific delivered models are consistently ranked the same way regardless of test-set sampling noise | The *model-training process* for GB is not reliably better than LR's — the Week 6 result may reflect a lucky training split, not a real GB advantage |

**Neither result is wrong. They're not measuring the same thing.** My
test is reassuring about the artifacts already in this repo, in the
narrow sense that GB is not an unstable model living rooted in a specific
lucky test partition. Data Science's test, if it retrained per seed, is
answering the more important question for a *final recommendation*:
whether GB's advantage generalizes when the whole pipeline (data split →
feature engineering → fit) is rerun — which is what actually matters if
GB or LR gets retrained again in the future.

## What needs to happen before either conclusion is used going forward

**Sent back to Data Science (see updated `week7_ds_feedback.md`):** a
direct question — did the multi-seed test retrain both models on each
seed's training split, or evaluate the same fixed fitted models across
different test partitions? The retraction is only warranted under the
first interpretation. If it was actually the second (same fixed models,
different test splits), then my result (5/5, deterministic) should have
been theirs too, and the discrepancy would need a different explanation
entirely — worth ruling out before treating either number as final.

## What did NOT change in this pipeline as a result

Per the same logic as before: `predict()`'s dispatcher still makes no
assumption about which model is "better," so this open question doesn't
block anything currently in the pipeline. It does mean the model card's
retraction language should be marked as "reported, pending methodology
confirmation" rather than a settled fact — updated accordingly.
