from pathlib import Path
import requests

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

YEAR = 2025
MONTHS = range(1, 13)  # Jan–Dec
TRIP_TYPE = "yellow_tripdata"


def build_url(year: int, month: int) -> str:
    month_str = f"{month:02d}"
    filename = f"{TRIP_TYPE}_{year}-{month_str}.parquet"
    return f"{BASE_URL}/{filename}"


def download_file(url: str, output_path: Path):
    if output_path.exists():
        print(f"[SKIP] {output_path.name}")
        return

    print(f"[DOWNLOADING] {output_path.name}")
    r = requests.get(url, stream=True, timeout=60)

    if r.status_code != 200:
        print(f"[NOT FOUND] {url}")
        return

    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def main():
    output_dir = Path("data/raw/tlc")
    output_dir.mkdir(parents=True, exist_ok=True)

    for month in MONTHS:
        url = build_url(YEAR, month)
        out_file = output_dir / f"{TRIP_TYPE}_{YEAR}-{month:02d}.parquet"
        download_file(url, out_file)


if __name__ == "__main__":
    main()
