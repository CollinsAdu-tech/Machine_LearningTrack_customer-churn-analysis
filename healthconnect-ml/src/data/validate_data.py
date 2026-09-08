"""
HealthConnect ML Pipeline — Data Validation
Week 5 — Machine Learning Engineering Track

Stage 2 (Data Validation) of the ML workflow (Week 4 ML System Design Doc),
verified against the real 5,000-row dataset and the Data Science Week 5
handoff (Section 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

EXPECTED_COLUMNS = [
    "appointment_id",
    "patient_id",
    "gender",
    "age",
    "age_group",
    "appointment_type",
    "booking_date",
    "appointment_date",
    "appointment_day",
    "appointment_time",
    "booking_lead_days",
    "previous_appointments",
    "previous_no_shows",
    "reminder_sent",
    "reminder_channel",
    "distance_to_clinic_km",
    "waiting_time_minutes",
    "appointment_outcome",
]

VALID_APPOINTMENT_OUTCOMES = {"Attended", "No-Show", "Cancelled"}
VALID_REMINDER_SENT = {"Yes", "No"}
VALID_APPOINTMENT_TIMES = {"Morning", "Afternoon", "Evening"}

# Missingness expected per the Week 4 design doc / DS handoff Section 1 —
# flagged, not treated as a critical failure.
EXPECTED_MISSING_COLUMNS = {
    "reminder_channel",
    "distance_to_clinic_km",
    "waiting_time_minutes",
}


class DataValidationError(Exception):
    """Raised when the dataset fails a critical (non-recoverable) check."""


@dataclass
class ValidationReport:
    n_rows: int = 0
    n_columns: int = 0
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    missing_value_counts: dict[str, int] = field(default_factory=dict)
    duplicate_appointment_ids: int = 0
    duplicate_rows: int = 0
    invalid_outcome_values: int = 0
    invalid_reminder_sent_values: int = 0
    invalid_appointment_time_values: int = 0
    invalid_dates: int = 0
    no_show_gt_appointments_count: int = 0
    negative_value_issues: dict[str, int] = field(default_factory=dict)
    passed: bool = True
    critical_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "HealthConnect Data Validation Report",
            "=" * 40,
            f"Rows: {self.n_rows} | Columns: {self.n_columns}",
            f"Duplicate appointment_id count: {self.duplicate_appointment_ids}",
            f"Fully duplicate rows: {self.duplicate_rows}",
            f"Invalid appointment_outcome values: {self.invalid_outcome_values}",
            f"Invalid reminder_sent values: {self.invalid_reminder_sent_values}",
            f"Invalid appointment_time values: {self.invalid_appointment_time_values}",
            f"Unparseable dates: {self.invalid_dates}",
            f"previous_no_shows > previous_appointments: {self.no_show_gt_appointments_count}",
            "",
            "Missing value counts:",
        ]
        for col, count in self.missing_value_counts.items():
            if count:
                lines.append(f"  - {col}: {count}")
        if self.negative_value_issues:
            lines.append("Negative-value issues:")
            for col, count in self.negative_value_issues.items():
                lines.append(f"  - {col}: {count}")
        if self.warnings:
            lines.append("\nWarnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        if self.critical_issues:
            lines.append("\nCRITICAL ISSUES:")
            lines.extend(f"  - {c}" for c in self.critical_issues)
        lines.append(f"\nOverall result: {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


def validate_dataset(df: pd.DataFrame, raise_on_critical: bool = False) -> ValidationReport:
    """
    Run schema and data-quality checks on the raw appointment dataframe.

    Implements the checks in Section 4 (Stage 2) of the Week 4 ML System
    Design Document: schema, missing values, duplicates, invalid categorical
    values, invalid dates, and historical inconsistencies.
    """
    report = ValidationReport(n_rows=len(df), n_columns=df.shape[1])

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    unexpected_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    report.missing_columns = missing_cols
    report.unexpected_columns = unexpected_cols

    if missing_cols:
        msg = f"Missing expected columns: {missing_cols}"
        report.critical_issues.append(msg)
        report.passed = False
        if raise_on_critical:
            raise DataValidationError(msg)
        return report

    if unexpected_cols:
        report.warnings.append(f"Unexpected extra columns found: {unexpected_cols}")

    missing_counts = df.isna().sum()
    report.missing_value_counts = {
        col: int(count) for col, count in missing_counts.items() if count > 0
    }
    unexpected_missing = set(report.missing_value_counts) - EXPECTED_MISSING_COLUMNS
    if unexpected_missing:
        report.warnings.append(
            f"Missing values found in columns not flagged in Week 4 design: {sorted(unexpected_missing)}"
        )

    report.duplicate_appointment_ids = int(df["appointment_id"].duplicated().sum())
    report.duplicate_rows = int(df.duplicated().sum())
    if report.duplicate_appointment_ids:
        report.warnings.append(
            f"{report.duplicate_appointment_ids} duplicate appointment_id values found."
        )

    report.invalid_outcome_values = int(
        (~df["appointment_outcome"].isin(VALID_APPOINTMENT_OUTCOMES)).sum()
    )
    report.invalid_reminder_sent_values = int(
        (~df["reminder_sent"].isin(VALID_REMINDER_SENT)).sum()
    )
    report.invalid_appointment_time_values = int(
        (~df["appointment_time"].isin(VALID_APPOINTMENT_TIMES)).sum()
    )
    if report.invalid_outcome_values:
        report.warnings.append(
            f"{report.invalid_outcome_values} rows have an appointment_outcome outside "
            f"{VALID_APPOINTMENT_OUTCOMES}."
        )

    booking_dates = pd.to_datetime(df["booking_date"], errors="coerce")
    appt_dates = pd.to_datetime(df["appointment_date"], errors="coerce")
    report.invalid_dates = int(booking_dates.isna().sum() + appt_dates.isna().sum())
    if report.invalid_dates:
        report.warnings.append(
            f"{report.invalid_dates} unparseable date values across booking_date/appointment_date."
        )
    if not report.invalid_dates:
        report.warnings.append(
            "booking_date/appointment_date parse successfully but are stored as "
            "M/D/YYYY text, not the ISO format stated in the data dictionary. "
            "Downstream code should not assume ISO format."
        )

    inconsistent = df["previous_no_shows"] > df["previous_appointments"]
    report.no_show_gt_appointments_count = int(inconsistent.sum())
    if report.no_show_gt_appointments_count:
        report.warnings.append(
            f"{report.no_show_gt_appointments_count} rows have previous_no_shows > "
            "previous_appointments (logically inconsistent)."
        )

    numeric_nonneg_cols = [
        "age",
        "booking_lead_days",
        "previous_appointments",
        "previous_no_shows",
        "distance_to_clinic_km",
        "waiting_time_minutes",
    ]
    for col in numeric_nonneg_cols:
        neg_count = int((df[col] < 0).sum())
        if neg_count:
            report.negative_value_issues[col] = neg_count
            report.warnings.append(f"{neg_count} negative values found in {col}.")

    implausible_age = int(((df["age"] < 0) | (df["age"] > 110)).sum())
    if implausible_age:
        report.warnings.append(f"{implausible_age} rows have an implausible age value.")

    if report.n_rows == 0:
        report.critical_issues.append("Dataset has zero rows after load.")
        report.passed = False

    if report.critical_issues and raise_on_critical:
        raise DataValidationError("; ".join(report.critical_issues))

    if report.critical_issues:
        report.passed = False

    return report
