from __future__ import annotations

import pandas as pd
import h3
from typing import Dict


REQUIRED_COLUMNS = {
    "date",
    "hour",
    "h3_cell",
    "trip_count",
}


def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_no_nulls(df: pd.DataFrame) -> None:
    nulls = df[REQUIRED_COLUMNS].isnull().any()
    bad_cols = nulls[nulls].index.tolist()
    if bad_cols:
        raise ValueError(f"Null values found in columns: {bad_cols}")


def validate_h3_resolution(df: pd.DataFrame, expected_res: int) -> None:
    sample_cells = df["h3_cell"].dropna().unique()[:100]

    bad = []
    for cell in sample_cells:
        if h3.get_resolution(cell) != expected_res:
            bad.append(cell)

    if bad:
        raise ValueError(
            f"H3 resolution mismatch. Expected={expected_res}, "
            f"Bad cells sample={bad[:5]}"
        )


def validate_trip_counts(df: pd.DataFrame) -> None:
    if (df["trip_count"] <= 0).any():
        raise ValueError("Found non-positive trip counts")


def validation_summary(df: pd.DataFrame) -> Dict[str, int]:
    return {
        "rows": len(df),
        "unique_h3_cells": df["h3_cell"].nunique(),
        "total_trips": int(df["trip_count"].sum()),
        "days": df["date"].nunique(),
    }
