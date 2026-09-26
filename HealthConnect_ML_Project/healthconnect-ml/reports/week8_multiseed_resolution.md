# Week 8 — Multi-Seed Discrepancy: Resolved

## Resolution

Data Science confirmed their multi-seed test **retrained both models from
scratch on each seed's training split** (fixed hyperparameters, varying
only the split), while this pipeline's Week 7 re-test evaluated the same
**already-fitted, fixed artifacts** across different test partitions —
exactly the methodological difference hypothesized in
`week7_multiseed_verification.md`.

## Independent confirmation (not just accepted)

Data Science's response included a demonstration: resampling the same
fixed seed-42 test set 5 times without retraining, showing GB winning
5/5. Rather than take that at face value, it was reproduced here
independently, using the same `.joblib` artifacts already in this repo:

```
Resample 1: AUC_LR=0.6621  AUC_GB=0.6899  GB wins
Resample 2: AUC_LR=0.6865  AUC_GB=0.6982  GB wins
Resample 3: AUC_LR=0.6926  AUC_GB=0.6958  GB wins
Resample 4: AUC_LR=0.6914  AUC_GB=0.6968  GB wins
Resample 5: AUC_LR=0.6735  AUC_GB=0.6877  GB wins
```

**GB wins 5/5**, with AUC values in the same range Data Science reported.
This confirms their explanation rather than just trusting it.

## What this means, precisely

Two different, both-true findings coexist:

1. **On the actual available data** (the fixed test population these
   models were evaluated against), GB reliably and measurably outperforms
   LR — confirmed independently, not sampling noise.
2. **If the entire pipeline were retrained on a different random slice of
   patients**, GB's advantage is not guaranteed — 2 of 5 fresh-trained
   comparisons favored GB, 3 favored LR.

Neither the Week 6 "GB is better" claim nor the Week 7 "it's a coin flip"
claim was wrong. They answer different questions. The final model
decision (below) is based on claim 1, which is the one that applies to
the actual deployed artifact — not a hypothetical retrain.

## Status

**Closed.** No further action needed on this specific question. See
`docs/model_card.md` for the finalized model card and
`reports/week8_final_model_decision.md` for what this means for the
final pipeline.
