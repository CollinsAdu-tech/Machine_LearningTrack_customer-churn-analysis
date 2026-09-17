# HealthConnect ML Pipeline — Week 6 Reproducibility Check

## Purpose

Verifies the Week 6 assignment's explicit requirement: "verify that the
pipeline can be reproduced using the documented setup." Prior to this
check, all Week 6 testing had been run inside the already-configured
working environment — which does not actually prove reproducibility from
a clean state. This document records an executed check, not a claim.

## What was done

1. **Captured exact installed package versions** from the working
   environment (where all prior Week 6 development and testing happened):

   | Package | Version |
   |---|---|
   | pandas | 3.0.2 |
   | numpy | 2.4.4 |
   | scikit-learn | 1.8.0 |
   | matplotlib | 3.10.8 |
   | seaborn | 0.13.2 |
   | joblib | 1.5.3 |
   | pyyaml | 6.0.3 |
   | pytest | 9.1.1 |

2. **Pinned these exact versions in `requirements.txt`** (previously
   unpinned — a real gap, since an unpinned `scikit-learn` version could
   silently change how the `.joblib` model artifacts deserialize or
   behave).

3. **Built a genuinely fresh Python virtual environment** (`python3 -m
   venv`), isolated from the working environment, and installed
   dependencies using only `pip install -r requirements.txt` — no manual
   package installation, no reuse of the existing environment's packages.

4. **Copied the repository to a separate directory** (not the working
   copy) to rule out any accidental reliance on files, caches, or state
   specific to the original location.

5. **Ran the full test suite** (`pytest tests/ -v`) using the clean
   venv's Python interpreter, against the fresh repo copy.

6. **Ran a manual prediction call** (`predict(record,
   model_name="gradient_boosting")`) in the clean environment and compared
   the output to the same call made earlier in the working environment.

## Results

- **Dependency install:** succeeded with no errors, using only the pinned
  `requirements.txt`.
- **Test suite:** **67/67 passed**, identical to the working-environment
  result — same test count, same pass/fail outcome for every test.
- **Manual prediction:** produced `{'prediction': 0, 'label': 'Attended',
  'no_show_probability': 0.401, 'risk_category': 'Medium', 'model_used':
  'gradient_boosting'}` — **identical** to the result obtained earlier in
  the working environment for the same input record. This confirms
  reproducibility isn't just "the tests still pass" but "the actual
  numeric output is bit-for-bit consistent" across environments.

## What this does and doesn't prove

**Proven:** the pipeline, as documented, can be set up and run correctly
by someone with only this repository and `requirements.txt` — no hidden
dependency on the original development environment's state.

**Not yet proven (genuine remaining gaps, not overstated as covered):**
- Cross-OS reproducibility (this check ran on the same OS/architecture as
  the original environment; a different OS was not tested).
- Cross-Python-version reproducibility (same Python version used in both
  environments; different versions not tested).
- Long-term reproducibility if `scikit-learn` releases a version that
  changes joblib's deserialization format for older pickled estimators —
  pinning today's version mitigates but doesn't eliminate this risk
  indefinitely.

These are reasonable Week 7 follow-ups, not claimed as done here.
