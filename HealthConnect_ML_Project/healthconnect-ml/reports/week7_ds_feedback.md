# Feedback to Data Science — Week 7 Multi-Seed Verification

**From:** ML Engineering
**To:** Data Science
**Re:** Independent re-test of the multi-seed retraction

---

## Summary

Before updating the pipeline's documentation to reflect the GB
retraction, I re-ran the same 5-seed comparison myself against the actual
`.joblib` artifacts already integrated in this pipeline
(`healthconnect_logreg_pipeline.joblib`, `healthconnect_gb_pipeline.joblib`).

**It didn't reproduce your reported numbers**, and I think I've found
why — but I need you to confirm before either of us treats one result as
final.

## What I ran

Same seeds you reported (42, 7, 123, 2024, 99), same
`GroupShuffleSplit(test_size=0.2, random_state=seed)` grouped by
`patient_id`, evaluated against the two artifacts I already have on file.

## What I got

| Seed | Your reported gap (GB−LR) | My gap (GB−LR) |
|---|---:|---:|
| 42 | +0.0116 | +0.0116 ✅ |
| 7 | +0.0015 | +0.0152 |
| 123 | −0.0028 | +0.0169 |
| 2024 | −0.0127 | +0.0078 |
| 99 | −0.0077 | +0.0090 |

**Yours:** mean −0.002, GB wins 2/5.
**Mine:** mean +0.0121, **GB wins 5/5.**

Only seed 42 matches — which makes sense, since that's the exact split
both our Week 6 evaluations used.

## Why I think this happened

I checked whether the GB artifact itself has any randomness that survives
fitting: `subsample=1.0`, `max_features=None`, fixed `random_state=42`.
That means the delivered GB pipeline is **fully deterministic** —
`predict_proba()` on the same `X` always returns the same output. My test
could only vary one thing across the 5 runs: *which patients ended up in
the test set*. The model itself never changed.

For your test to produce results that flip sign across seeds, I think
you must have **retrained both models from scratch on each seed's
training split** — which is a completely different (and honestly, more
important) question than the one I just answered. Retraining introduces
real variation (different tree structures, different splits learned);
just resampling the test set on a fixed model doesn't.

## The question I need answered

**Did your multi-seed test retrain both models per seed, or evaluate the
same fixed, already-fitted artifacts across 5 different test splits?**

If you retrained: your retraction stands, and it's telling us something
important — that the *training process* for GB isn't reliably better
than LR's, even though the specific artifacts you sent me happen to be a
good pair (GB consistently beats LR on those two, regardless of test-set
sampling — that's a real, useful, separate fact).

If you evaluated the same fixed artifacts: then my result should have
matched yours exactly (since there's no randomness left to differ on),
and we'd need to figure out where the actual discrepancy is coming from
— possibly a difference in how `historical_no_show_rate` was computed, a
different `GroupShuffleSplit` scikit-learn version behavior, or something
else entirely worth chasing down.

## What I've done in the meantime

- Documented both result sets and this open question in the model card
  and pipeline reports — not asserting either is final.
- Made no pipeline code changes either way — `predict()` already treats
  both models as fully interchangeable, so this doesn't block anything
  operationally regardless of the answer.
- Full technical writeup with the deterministic-parameter check:
  `reports/week7_multiseed_verification.md`, if useful.

Let me know which it was when you get a chance — this is the one thing
actually blocking a confident final model recommendation.

Thanks!
