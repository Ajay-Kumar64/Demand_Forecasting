"""
Batch inference: score a date range with trained artifacts.

Run:  python -m scripts.batch_predict \
        --demand-parquet data/feature_store/h3_demand_2025.parquet \
        --models-dir artifacts/models \
        --start 2025-11-24T00:00 --end 2025-11-25T00:00 \
        --output artifacts/predictions/predictions.parquet
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modeling.train import (  # add predict_quantiles
    load_sparse_demand, build_grid, build_feature_layers,
    build_matrix, predict_quantiles, slot_bucket, LAG_HISTORY_SLOTS,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demand-parquet", default="data/feature_store/h3_demand_2025.parquet")
    ap.add_argument("--models-dir", default="artifacts/models")
    ap.add_argument("--start", required=True, help="ISO timestamp, e.g. 2025-11-24T00:00")
    ap.add_argument("--end", required=True, help="ISO timestamp (exclusive)")
    ap.add_argument("--output", default="artifacts/predictions/predictions.parquet")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    meta = json.loads((models_dir / "train_meta.json").read_text())
    with open(models_dir / "lgbm_quantile_models.pkl", "rb") as f:
        models = pickle.load(f)
    cell_mean = np.load(models_dir / "cell_mean.npy")
    profile = np.load(models_dir / "profile_dowhour.npy")
    cells = np.array(json.loads((models_dir / "cells.json").read_text()))

    sparse = load_sparse_demand(Path(args.demand_parquet))
    D, _, ts_slots = build_grid(sparse)
    del sparse
    if D.shape[0] != meta["n_slots"]:
        raise RuntimeError("Grid changed since training — re-run src.modeling.train first.")

    start_i = int((pd.Timestamp(args.start) - ts_slots[0])
                  // pd.Timedelta(minutes=15))
    end_i = int((pd.Timestamp(args.end) - ts_slots[0]) // pd.Timedelta(minutes=15))
    if start_i < LAG_HISTORY_SLOTS:
        raise ValueError(f"--start must be >= {ts_slots[LAG_HISTORY_SLOTS]} "
                         f"(feature warm-up window)")
    end_i = min(end_i, D.shape[0])

    F, slotvec = build_feature_layers(D, ts_slots)
    F["cell_dowhour_mean"] = profile[slot_bucket(ts_slots), :]
    n_cells = D.shape[1]

    X = build_matrix(F, slotvec, cell_mean, start_i, end_i, n_cells)
    q10, q50, q90 = predict_quantiles(models, X)    
    n_range = end_i - start_i
    out = pd.DataFrame({
        "h3_cell": np.tile(cells, n_range),
        "ts_15min": np.repeat(ts_slots[start_i:end_i], n_cells),
        "q10": q10, "q50": q50, "q90": q90,
        "conf_width": q90 - q10,
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"[SAVED] {len(out):,} predictions -> {args.output}")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
