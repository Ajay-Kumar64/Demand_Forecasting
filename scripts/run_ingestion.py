import yaml
from pathlib import Path
from src.ingestion.tlc_ingest import ingest_csv

CONFIG_PATH = "configs/data.yaml"


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    raw_path = Path(cfg["raw_data"]["taxi_path"])
    output_path = Path(cfg["processed_data"]["output_path"])

    output_path.mkdir(parents=True, exist_ok=True)

    checksums = {}

    for csv_file in raw_path.glob("*.csv"):
        print(f"Ingesting {csv_file.name}")
        checksum = ingest_csv(csv_file, output_path)
        checksums[csv_file.name] = checksum

    print("Ingestion completed")
    print("Checksums:", checksums)


if __name__ == "__main__":
    main()
