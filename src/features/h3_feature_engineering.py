import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from typing import List


def add_lags(df: pd.DataFrame, lag_hours: List[int], freq_minutes: int = 15) -> pd.DataFrame:
    """
    Add lag features per h3_cell.
    lag_hours: list of hours to lag (e.g., [1, 4, 168])
    freq_minutes: frequency of buckets, default 15
    """
    df = df.sort_values(["h3_cell", "ts_15min"])
    for lag_h in lag_hours:
        lag_periods = lag_h * 60 // freq_minutes
        df[f"demand_t-{lag_h}h"] = df.groupby("h3_cell")["demand"].shift(lag_periods)
    return df


def add_rolling_stats(df: pd.DataFrame, windows_hours: List[int], freq_minutes: int = 15) -> pd.DataFrame:
    """
    Add rolling mean and variance features per h3_cell
    windows_hours: list of rolling windows in hours (e.g., [3,6])
    """
    df = df.sort_values(["h3_cell", "ts_15min"])
    for win_h in windows_hours:
        window = win_h * 60 // freq_minutes
        df[f"demand_roll_mean_{win_h}h"] = df.groupby("h3_cell")["demand"].transform(lambda x: x.shift(1).rolling(window).mean())
        df[f"demand_roll_var_{win_h}h"] = df.groupby("h3_cell")["demand"].transform(lambda x: x.shift(1).rolling(window).var())
    return df


def add_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical encoding for hour-of-day and day-of-week
    """
    df["hour"] = df["ts_15min"].dt.hour
    df["day_of_week"] = df["ts_15min"].dt.dayofweek

    # Hour-of-day
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Day-of-week
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add US federal holiday and weekend flags
    """
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df["ts_15min"].min(), end=df["ts_15min"].max())
    df["is_holiday"] = df["ts_15min"].dt.normalize().isin(holidays).astype(int)
    df["is_weekend"] = df["ts_15min"].dt.dayofweek.isin([5, 6]).astype(int)
    return df
