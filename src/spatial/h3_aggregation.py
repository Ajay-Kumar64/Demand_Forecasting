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
    df["ts_15min"] = (
        pd.to_datetime(df["tpep_pickup_datetime"])
        .dt.floor("15min")
    )
    return df


def add_h3_from_zone(df: pd.DataFrame, zone_lookup: dict) -> pd.DataFrame:
    df["h3_cell"] = df["PULocationID"].map(zone_lookup)
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
