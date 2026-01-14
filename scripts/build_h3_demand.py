from pathlib import Path
import pandas as pd

from src.spatial.h3_aggregation import process_parquet, write_feature_store
from src.spatial.taxi_zone_lookup import build_zone_h3_lookup


RAW_PATH = Path("data/raw/tlc")
ZONE_SHP = Path("data/raw/taxi_zones/taxi_zones.shp")
OUTPUT_PATH = Path("data/feature_store/h3_demand_2025.parquet")


def main():
    zone_lookup = build_zone_h3_lookup(ZONE_SHP)

    all_frames = []

    for parquet_file in sorted(RAW_PATH.glob("yellow_tripdata_2025-*.parquet")):
        print(f"[PROCESSING] {parquet_file.name}")
        df_agg = process_parquet(parquet_file, zone_lookup)
        all_frames.append(df_agg)

    if not all_frames:
        raise RuntimeError("No parquet files processed")

    final_df = (
        pd.concat(all_frames, ignore_index=True)
        .sort_values(["h3_cell", "ts_15min"])
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_feature_store(final_df, OUTPUT_PATH)

    print(f"[SUCCESS] Feature store written → {OUTPUT_PATH}")
    print(f"Rows: {len(final_df):,}")


if __name__ == "__main__":
    main()
