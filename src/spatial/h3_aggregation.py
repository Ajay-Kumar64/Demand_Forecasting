from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "PULocationID"
]


def validate_columns(df: pd.DataFrame):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_time_bucket(df: pd.DataFrame) -> pd.DataFrame:
    # TLC pickups are local wall-clock time. Nov fall-back makes 1:00-1:59 AM
    # occur twice; naive flooring double-counts it. Localize -> convert to
    # UTC -> drop tz -> floor, so every 15-min slot is unique and true.
    df = df.sort_values("tpep_pickup_datetime")
    ts = pd.to_datetime(df["tpep_pickup_datetime"]).dt.tz_localize(
        "America/New_York", ambiguous="infer", nonexistent="shift_forward"
    )
    df["ts_15min"] = (
        ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("15min")
    )
    return df


def add_h3_from_zone(df: pd.DataFrame, zone_lookup: dict) -> pd.DataFrame:
    mapped = df["PULocationID"].map(zone_lookup)
    n_dropped = int(mapped.isna().sum())
    if n_dropped:
        pct = n_dropped / len(df) * 100
        top = df.loc[mapped.isna(), "PULocationID"].value_counts().head(5).to_dict()
        print(f"[DATA-QUALITY] dropping {n_dropped:,}/{len(df):,} rows ({pct:.2f}%) "
              f"with unmapped PULocationID (264/265 etc.); top: {top}")
    df["h3_cell"] = mapped
    return df.dropna(subset=["h3_cell"])


def aggregate_demand(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["h3_cell", "ts_15min"])
        .size()
        .reset_index(name="demand")
    )


def process_parquet(file_path: Path, zone_lookup: dict) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    validate_columns(df)

    df = add_time_bucket(df)
    df = add_h3_from_zone(df, zone_lookup)

    return aggregate_demand(df)


def write_feature_store(df: pd.DataFrame, output_path: Path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
