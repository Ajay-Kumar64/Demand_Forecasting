from pathlib import Path
import pandas as pd
import hashlib
import pyarrow.parquet as pq
import pyarrow as pa

REQUIRED_COLUMNS = {
    "tpep_pickup_datetime": "pickup_datetime",
    "pickup_longitude": "pickup_longitude",
    "pickup_latitude": "pickup_latitude"
}


def file_checksum(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def validate_schema(df: pd.DataFrame):
    missing = set(REQUIRED_COLUMNS.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=REQUIRED_COLUMNS)


def ingest_csv(csv_path: Path, output_dir: Path):
    df = pd.read_csv(csv_path)

    validate_schema(df)
    df = normalize_columns(df)

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df = df.dropna(subset=["pickup_latitude", "pickup_longitude"])

    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month

    table = pa.Table.from_pandas(df)

    pq.write_to_dataset(
        table,
        root_path=output_dir,
        partition_cols=["year", "month"]
    )

    return file_checksum(csv_path)
