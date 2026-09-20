# Week 7 Project Summary — Machine Learning Engineering Track

**1. What I planned to test:**
The integrated pipeline's handling of invalid/edge-case input (Week 7
task #9), and validate whether Data Science's Week 7 model-selection
retraction held up before updating pipeline documentation to reflect it.

**2. What I actually tested:**
- Six invalid-input scenarios (null required fields, wrong-type values,
  a logically-impossible field combination, negative age, empty-string
  categoricals, extreme distance values) across all three routable
  models (baseline, LR candidate, GB candidate) — 18 tests total.
- Independently re-ran Data Science's reported 5-seed AUC stability
  comparison against the actual delivered model artifacts.

**3. The most important testing results:**
- Found and fixed 3 real bugs in prediction-time input validation.
- Found that Data Science's multi-seed retraction (GB wins 2/5 seeds)
  did not reproduce against the actual artifacts (my re-test: GB wins
  5/5) — traced to a likely methodological difference (retrain-per-seed
  vs. evaluate-fixed-artifacts-per-seed), not an error on either side.

**4. Issues or weaknesses identified:**
- Null values for required fields were silently accepted and reached the
  model with no warning.
- Wrong-type values (e.g. a string where a number was expected) caused a
  raw, unhelpful sklearn error instead of a clean rejection.
- A logical-consistency rule already enforced on training data
  (`previous_no_shows` cannot exceed `previous_appointments`) was not
  enforced on live prediction requests.
- Data Science's Week 7 multi-seed conclusion could not be independently
  reproduced against the actual delivered artifacts.

**5. Improvements/refinements made:**
- Added `find_null_fields()` and `find_non_numeric_fields()` to
  `model_utils.py`, wired into both `predict.py` and
  `candidate_models.py`.
- Added an explicit `previous_no_shows > previous_appointments` check at
  prediction time, closing the gap between training-time and
  prediction-time validation.
- Updated `docs/model_card.md` and related reports to present the
  multi-seed discrepancy transparently rather than silently adopting
  Data Science's retraction as fact.

**6. Retesting results:**
All 18 new edge-case tests pass after the fixes. Full suite: **85/85
passing.** The multi-seed re-test is deterministic and reproducible —
re-running it against the same artifact files always returns the same
result (GB 5/5, mean +0.0121).

**7. Which track(s) I collaborated with:**
Data Science.

**8. What was tested collaboratively:**
Data Science's Week 7 multi-seed stability claim, tested independently
against the actual model artifacts already integrated into this
pipeline — not just discussed or accepted as reported.

**9. What changed as a result:**
The model card and related documentation now present both result sets
(Data Science's reported numbers and this pipeline's independent
re-test) with a specific, identified reason for the divergence, and a
concrete clarifying question was sent back to Data Science rather than
either silently trusting or silently overriding their conclusion.

**10. Key findings or validation outcomes:**
- The pipeline's existing model-routing design (built in Week 6, no
  hardcoded model preference) meant the entire GB retraction required
  zero code changes — only documentation updates. Good foundational
  design paid off here.
- The two specific model artifacts already deployed in this pipeline are
  fully deterministic and, on this pipeline's own test, consistently
  rank GB above LR — a fact independent of whichever broader
  training-stability conclusion Data Science's methodology supports.

**11. Major challenges encountered:**
Diagnosing *why* two seemingly identical tests (same seeds, same
methodology description, same metric) produced contradictory results
required inspecting the fitted model's hyperparameters directly rather
than assuming either side had made an error — the deterministic-artifact
finding was the key that made the discrepancy explicable rather than
just confusing.

**12. Important decisions made:**
Chose not to simply adopt Data Science's retraction into the
documentation at face value, despite it being the more recent and
seemingly more thorough analysis — because this project's standard
(established in Week 6) is to verify claims against actual artifacts
before writing them into permanent documentation, and that standard
doesn't get relaxed just because a claim happens to be a retraction
rather than a positive result.

**13. Remaining limitations:**
The actual cause of the discrepancy (retrain-per-seed vs.
fixed-artifact-per-seed) is inferred from the GB model's hyperparameters,
not yet confirmed by Data Science directly. Until confirmed, neither the
Week 6 "GB is better" nor the Week 7 "it's a coin flip" conclusion should
be treated as final for a production model-selection decision.

**14. Remaining issues or dependencies:**
- Awaiting Data Science's confirmation of their multi-seed methodology.
- Three known, lower-severity input-validation gaps documented but not
  fixed this pass (negative age, empty-string categoricals, extreme
  distance values) — see `reports/week7_pipeline_test_results.md`.
- The two Week 5-inherited conditional-feature assumptions (reminder
  timing, historical-count construction) remain unresolved, still
  pending Project Management escalation per Data Science.

**15. Contribution to the overall HealthConnect project:**
Prevented an unverified claim from propagating into permanent project
documentation and applied the same evidentiary standard to a retraction
as would be applied to any other reported result — directly relevant
given the project's overall reliance on trustworthy, evidence-based
decisions across tracks.

**16. What must be completed before Week 8:**
- Data Science's confirmation of the multi-seed methodology question.
- A decision (informed by whichever multi-seed interpretation is
  confirmed correct) on how or whether to surface a model-confidence
  signal for the two identified weak segments (age 65+, Specialist
  Consultation).
- Group-aware split fix for the Week 5 baseline, so all three models are
  finally comparable on identical footing (carried over from Week 6).
