import sys
import pandas as pd


def main(path: str, date: str):
    df = pd.read_parquet(path)

    # Convert timestamp to datetime if not already
    df["ts_15min"] = pd.to_datetime(df["ts_15min"])

    # Derive a 'date' column
    df["date"] = df["ts_15min"].dt.date

    # Filter for the requested date
    df_day = df[df["date"] == pd.to_datetime(date).date()]

    if df_day.empty:
        raise ValueError(f"No data found for date {date}")

    print(f"[DATE] {date}")
    print("Rows:", len(df_day))
    print("Total trips:", int(df_day["demand"].sum()))
    print("Unique H3 cells:", df_day["h3_cell"].nunique())
    print("\nTop 10 demand cells:")
    print(
        df_day.sort_values("demand", ascending=False)
        .head(10)[["h3_cell", "demand"]]
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise RuntimeError(
            "Usage: python -m scripts.smoke_test_one_day <parquet_path> <YYYY-MM-DD>"
        )

    main(sys.argv[1], sys.argv[2])
