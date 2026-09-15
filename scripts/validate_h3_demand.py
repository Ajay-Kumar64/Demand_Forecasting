import argparse
import pandas as pd

from src.validation.h3_demand_validator import (
    validate_schema,
    validate_no_nulls,
    validate_h3_resolution,
    validate_demand_values,      # was validate_trip_counts
    validation_summary,
)


def main(path: str, h3_res: int):
    print(f"[LOAD] {path}")
    df = pd.read_parquet(path)

    print("[VALIDATE] schema")
    validate_schema(df)

    print("[VALIDATE] nulls")
    validate_no_nulls(df)

    print("[VALIDATE] demand values")
    validate_demand_values(df)   # was validate_trip_counts(df)

    print("[VALIDATE] h3 resolution")
    validate_h3_resolution(df, expected_res=h3_res)

    summary = validation_summary(df)
    print("[SUCCESS] Validation passed")
    print("[SUMMARY]")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        required=True,
        help="Path to H3 demand parquet",
    )
    parser.add_argument(
        "--h3-res",
        type=int,
        required=True,
        help="Expected H3 resolution",
    )

    args = parser.parse_args()
    main(args.path, args.h3_res)
