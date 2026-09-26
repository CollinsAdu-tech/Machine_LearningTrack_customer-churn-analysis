# Week 8 — Final Model Decision & Pipeline Finalization

## Final model: Gradient Boosting

**Artifact:** `models/candidates/healthconnect_gb_pipeline.joblib` — the
exact file already in this repo since Week 6. No new artifact was
delivered; none was needed.

**Basis for the decision, stated precisely (per Data Science):** on the
actual available test population, GB consistently and measurably
outperforms LR (confirmed independently above — 5/5 wins across
resampling, not noise). The defensible claim is narrower than "GB is a
better algorithm for this problem in general" — it is "on our available
data, this specific trained GB model outperforms this specific trained LR
model, consistently." That narrower claim is sufficient to finalize on,
and is the one recorded in the final model card.

**Verification status:** this exact artifact was already fully verified
in Week 6 (`reports/week6_model_verification.md`) — schema, reload-
identity, reproduced evaluation metrics, and the confirmed/fixed
`reminder_channel` null-handling bug. Since it's the same file, not a new
one, that verification stands and was not re-run. Confirmed with Data
Science that no new file was coming before relying on this.

## Feature schema: unchanged, explicitly confirmed

No changes since the Week 6 handoff. Same 6 numeric + 5 categorical
features, same one-hot category order, same preprocessing. No config or
code changes required.

## Final evaluation metrics (official test set, n=966)

| Metric | Value |
|---|---|
| Accuracy | 65.11% |
| ROC-AUC | 0.6941 |
| Recall (No-Show) | 66.7% |
| Confusion matrix | TN 307 / FP 176 / FN 161 / TP 322 |

**Honesty note carried into documentation, per Data Science's explicit
request:** across the 5 retrained-per-seed comparisons, GB's AUC ranged
0.648–0.704 (mean ≈0.677). 0.6941 is the number for *this specific
artifact*, not a guaranteed floor for any future retrain. Model card
states this explicitly rather than presenting 0.6941 as an unconditional
figure.

## What this model can and cannot be used for

Recorded near-verbatim from Data Science's final statement — see
`docs/model_card.md` for the full text. Key points carried through
exactly, not paraphrased into something that could drift from their
intent:
- Suitable for flagging elevated-risk appointments for optional
  preventive follow-up only
- Not suitable for automated decisions with direct patient consequences,
  clinical/diagnostic use, or generalization beyond this fictional dataset
- 37–46% of predictions fall in a genuinely uncertain zone (0.4–0.6)
- Materially weaker on two segments: age 65+ (AUC 0.664, Recall 0.595)
  and Specialist Consultation (AUC 0.616, Recall 0.553)
- Not fairness-tested beyond the segment check, not calibration-tested,
  not production-ready without further validation on real data

## The two previously-open conditional-feature assumptions: closed as documented assumptions

There is no live PM/business stakeholder in this exercise to formally
confirm these, so rather than carry them as indefinitely "open" into the
final presentation, Data Science closed them as **explicit, stated
assumptions** rather than confirmed facts:

- **Reminder timing:** assumed prediction occurs 24–48 hours before the
  appointment (after reminders are typically sent). Not independently
  verified against real clinic operations.
- **Historical-count construction:** assumed `previous_appointments` /
  `previous_no_shows` are computed strictly prior to and excluding the
  current appointment. Consistent with normal construction, not
  independently verified against the source system.

This distinction matters: "documented assumption" is a different, more
honest status than "resolved" or "still blocking" — it says plainly that
the pipeline works *if* these assumptions hold, without claiming they've
been proven. Reflected in `config.yaml` and `docs/model_card.md`.

## What was NOT re-verified this cycle, and why that's fine

- The GB artifact itself was not re-verified for schema/reload-identity,
  since it's confirmed to be the identical file already verified in
  Week 6 — re-running identical checks against an unchanged file would
  add no new information.
- Feature schema compatibility was not re-tested in code, since Data
  Science confirmed explicitly that nothing changed — this is recorded
  as their explicit confirmation, not an unstated assumption on this
  pipeline's part.

## Outstanding, optional follow-up (not required for Week 8)

Data Science offered to test whether re-tuning GB's hyperparameters
per-seed (rather than holding them fixed across the multi-seed test)
changes the retrain-stability picture. This is explicitly a "fair
follow-up," not a blocker — noted here so it isn't lost, but not treated
as required before finalizing Week 8.
