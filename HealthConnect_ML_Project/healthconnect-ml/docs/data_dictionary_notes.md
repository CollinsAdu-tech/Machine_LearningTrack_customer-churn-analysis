# HealthConnect — Data Dictionary Notes (ML Engineering)

Notes on the 18 columns in `HealthConnect_Appointment_Data.csv`, as verified
against the raw file (not just the data dictionary) during Week 5.

| Column | Type | ML Engineering treatment |
|---|---|---|
| `appointment_id` | text | Dropped — identifier, 5,000 unique values, 0 duplicates |
| `patient_id` | text | Dropped as a feature — 1,696 unique values, 1,394 patients appear more than once |
| `gender` | text | One-hot encoded — 3 categories |
| `age` | integer | Used directly (numeric, scaled) |
| `age_group` | text | Dropped — fully derivable from `age`, redundant |
| `appointment_type` | text | One-hot encoded — 4 categories |
| `booking_date` | text | Stored as `M/D/YYYY`, **not** ISO as the dictionary states. Parsed explicitly with `format="%m/%d/%Y"`; used only to derive `booking_lead_days` (verified consistent), then dropped |
| `appointment_date` | text | Same format note as above. Used only to derive/verify `appointment_day`, then dropped |
| `appointment_day` | text | One-hot encoded — verified to match the actual weekday of `appointment_date` in all 5,000 rows |
| `appointment_time` | text | One-hot encoded — 3 categories |
| `booking_lead_days` | integer | Used directly; verified to equal `appointment_date - booking_date` in all 5,000 rows |
| `previous_appointments` | integer | Used directly (conditional — see below) |
| `previous_no_shows` | integer | Used directly (conditional); verified never exceeds `previous_appointments` |
| `reminder_sent` | text | Used to derive `reminder_sent_flag` (conditional) |
| `reminder_channel` | text | 27.32% missing, but verified structural: every null exactly matches `reminder_sent == "No"`. Encoded as an explicit `"None"` category, not imputed |
| `distance_to_clinic_km` | float | 1.80% missing, median-imputed (fit on train fold only) |
| `waiting_time_minutes` | float | **Dropped — confirmed leakage.** Populated for 98.5% of No-Show/Cancelled rows, which shouldn't have a real waiting time |
| `appointment_outcome` | text | Target column. `Cancelled` rows (5.3%) excluded from the binary target pending business confirmation |

## Conditional features (open items)

Two groups of features are included in the current baseline but flagged as
conditional pending confirmation from the Data Science handoff:

1. `reminder_sent`, `reminder_channel`, `reminder_sent_flag` — depends on
   confirming when in the workflow the prediction is meant to run.
2. `previous_appointments`, `previous_no_shows`, and everything derived
   from them (`prior_no_show_rate`, `is_first_time_patient`) — depends on
   confirming these counts exclude the current appointment.

Both are listed separately in `configs/config.yaml` under
`conditional_features` so they can be removed in one place if either
assumption doesn't hold.
