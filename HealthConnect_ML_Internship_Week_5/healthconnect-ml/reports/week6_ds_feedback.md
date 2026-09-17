# Feedback to Data Science — Week 6 ML Engineering Integration Findings

**From:** ML Engineering
**To:** Data Science
**Re:** Integration of `healthconnect_logreg_pipeline.joblib` and
`healthconnect_gb_pipeline.joblib`

---

## Summary

Both artifacts were fully verified before integration. Every reported
number in your Week 6 deliverable (Section 5's Accuracy, ROC-AUC, and
confusion matrices for both models) was independently reproduced from the
raw CSV using your exact `GroupShuffleSplit(test_size=0.2,
random_state=42)` spec — exact match on every figure, both models. Reload
identity, one-hot category order, and unseen-category handling were also
confirmed as described.

One integration-relevant behavior was found that isn't covered in your
handoff and is worth flagging back to you, since it would affect anyone
else consuming these artifacts directly.

## Finding: `reminder_channel` null handling requires an upstream step not mentioned in the handoff

**What we found:** Your message describes both pipelines as working
directly on a raw DataFrame with "no separate encoding/scaling step
needed." That's accurate for encoding and scaling. It is not accurate for
one specific case: a raw `NaN` in `reminder_channel` (as produced by
`pandas.read_csv` on the original dataset) is not recognized by the
fitted `OneHotEncoder` as the `"None"` category. `handle_unknown='ignore'`
treats it exactly like a genuinely novel category — silently zeroing out
the entire `reminder_channel` one-hot block — rather than matching it to
`"None"`, which exists specifically to represent "no reminder was sent."

**Reproduction (either model, same result):**
```python
row_raw_nan = df[df['reminder_channel'].isna()].iloc[[0]]
row_explicit_none = row_raw_nan.copy()
row_explicit_none['reminder_channel'] = 'None'

pipe.predict_proba(row_raw_nan)[0, 1]        # 0.7759 (GB) / 0.7653 (LR)
pipe.predict_proba(row_explicit_none)[0, 1]  # 0.7891 (GB) / 0.7937 (LR)
```
No error, no warning — just a silently different, less-informed
prediction. This affects 1,366 of 5,000 rows (27.3%) in the raw dataset.

**Contrast case, confirmed NOT an issue:** `distance_to_clinic_km`'s raw
`NaN` values are handled correctly by your bundled `SimpleImputer` — no
concern there, mentioned only so it's clear we checked rather than
assumed.

**What we did on our end:** Added a mandatory preparation step
(`prepare_candidate_input()`) in our integration layer that converts
`reminder_channel` nulls to the literal string `"None"` before either
pipeline is called. This is now enforced structurally (every prediction
path routes through it) and pinned with an automated regression test
against both live artifacts.

**Suggested action on your end (optional, for your own documentation/testing):**
Your Week 7 testing requirement #5 ("unseen-category handling test") is
adjacent to this but doesn't quite cover it — an unseen category and a
missing value hit the same `handle_unknown='ignore'` code path but need
different handling upstream. Worth a one-line addition to your pipeline's
own documentation noting that `reminder_channel` nulls must be
pre-converted to `"None"` by any caller, not just us — otherwise this bug
will resurface for the next consumer of these artifacts who doesn't
happen to build the same preparation step we did.

## What we're returning to you (per the integration contract)

- **Integrated model interface:** `src/models/candidate_models.py`,
  callable as `predict_with_candidate(record, model_name)`.
- **Pipeline-compatible artifacts:** both `.joblib` files, unmodified,
  stored in `models/candidates/`.
- **Input/output contract:** documented in
  `reports/week6_model_verification.md`.
- **Validation checks + integration tests:** 21 tests in
  `tests/test_candidate_models.py`, including the pinned bug regression
  test above.
- **This feedback document**, closing the loop so the integration isn't
  one-directional.

## Not resolved by this integration (unchanged from your handoff)

The two Week 5-inherited open assumptions (reminder timing legitimacy,
historical-count construction) remain exactly as you left them — neither
model resolves these, and this integration doesn't attempt to.

---

*Evidence for every claim above: `reports/week6_model_verification.md`
and `tests/test_candidate_models.py` (run via `pytest
tests/test_candidate_models.py -v`).*
