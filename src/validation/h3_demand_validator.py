"""Data-quality checks for the H3 demand feature store."""
from __future__ import annotations

from typing import Dict

import h3
import pandas as pd

REQUIRED_COLUMNS = {"h3_cell", "ts_15min", "demand"}


def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_no_nulls(df: pd.DataFrame) -> None:
    nulls = df[list(REQUIRED_COLUMNS)].isnull().any()
    bad_cols = nulls[nulls].index.tolist()
    if bad_cols:
        raise ValueError(f"Null values found in columns: {bad_cols}")


def validate_h3_resolution(df: pd.DataFrame, expected_res: int) -> None:
    sample_cells = df["h3_cell"].dropna().unique()[:100]
    bad = [c for c in sample_cells if h3.get_resolution(c) != expected_res]
    if bad:
        raise ValueError(
            f"H3 resolution mismatch. Expected={expected_res}, "
            f"Bad cells sample={bad[:5]}"
        )


def validate_demand_values(df: pd.DataFrame) -> None:
    """Demand counts must be non-negative. Zero is VALID (empty cell-slots
    are the point of the zero-filled grid — the old validator rejected them)."""
    if (df["demand"] < 0).any():
        raise ValueError("Found negative demand values")


def validation_summary(df: pd.DataFrame) -> Dict:
    return {
        "rows": len(df),
        "unique_h3_cells": int(df["h3_cell"].nunique()),
        "total_demand": int(df["demand"].sum()),
        "zero_slot_share": round(float((df["demand"] == 0).mean()), 4),
        "t_min": str(pd.to_datetime(df["ts_15min"]).min()),
        "t_max": str(pd.to_datetime(df["ts_15min"]).max()),
    }
