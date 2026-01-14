import argparse
import pandas as pd
from src.features.h3_feature_engineering import (
    add_lags,
    add_rolling_stats,
    add_cyclical_time,
    add_calendar_features
)


def main(input_path: str, output_path: str):
    print(f"[LOAD] {input_path}")
    df = pd.read_parquet(input_path)

    print("[STEP] Add lag features")
    df = add_lags(df, lag_hours=[1, 4, 168])  # 1h, 1h, 1 week

    print("[STEP] Add rolling stats")
    df = add_rolling_stats(df, windows_hours=[3, 6])

    print("[STEP] Add cyclical time features")
    df = add_cyclical_time(df)

    print("[STEP] Add calendar features")
    df = add_calendar_features(df)

    print(f"[SAVE] {output_path}")
    df.to_parquet(output_path, index=False)
    print("[SUCCESS] Feature engineering complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input H3 demand parquet")
    parser.add_argument("--output", required=True, help="Output feature parquet")
    args = parser.parse_args()

    main(args.input, args.output)
