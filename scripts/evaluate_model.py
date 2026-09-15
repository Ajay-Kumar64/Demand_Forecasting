"""
Out-of-time evaluation against trained artifacts.

Evaluates ONLY the chronological holdout (never training data), reports the
verdict table (model vs 4 baselines on the identical window), pinball loss,
aggregate + conditional coverage, reliability, per-cell WAPE, and importance.

Run:  python -m scripts.evaluate_model \
        --demand-parquet data/feature_store/h3_demand_2025.parquet \
        --models-dir artifacts/models
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modeling.train import (  # add predict_quantiles to this import
    load_sparse_demand, build_grid, build_feature_layers, chronological_splits,
    build_matrix, predict_quantiles, slot_bucket, wape, pinball, FEATURE_NAMES,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demand-parquet", default="data/feature_store/h3_demand_2025.parquet")
    ap.add_argument("--models-dir", default="artifacts/models")
    ap.add_argument("--out-dir", default="artifacts/eval")
    args = ap.parse_args()

    models_dir, out_dir = Path(args.models_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((models_dir / "train_meta.json").read_text())
    with open(models_dir / "lgbm_quantile_models.pkl", "rb") as f:
        models = pickle.load(f)
    cell_mean = np.load(models_dir / "cell_mean.npy")
    profile = np.load(models_dir / "profile_dowhour.npy")
    cells = json.loads((models_dir / "cells.json").read_text())

    # rebuild the identical grid; abort if data changed since training
    sparse = load_sparse_demand(Path(args.demand_parquet))
    D, _, ts_slots = build_grid(sparse)
    del sparse
    if D.shape[0] != meta["n_slots"] or str(ts_slots[0])[:19] != str(meta["t0"])[:19]:
        raise RuntimeError("Grid changed since training — re-run src.modeling.train first.")

    F, slotvec = build_feature_layers(D, ts_slots)
    F["cell_dowhour_mean"] = profile[slot_bucket(ts_slots), :]   # train-derived lookup

    _, _, c2 = chronological_splits(D.shape[0])
    n_cells = D.shape[1]
    Xte = build_matrix(F, slotvec, cell_mean, c2, D.shape[0], n_cells)
    yte = D[c2:].ravel().astype(np.float32)
    n_test = D.shape[0] - c2

    p10, p50, p90 = predict_quantiles(models, Xte)
    print("\n================ SAME TEST WINDOW ================")
    print(f"Copy last 15min  WAPE: {wape(yte, F['demand_t-15m'][c2:].ravel())*100:.2f}%")
    print(f"Copy last hour   WAPE: {wape(yte, F['demand_t-1h'][c2:].ravel())*100:.2f}%")
    print(f"Copy last week   WAPE: {wape(yte, F['demand_t-168h'][c2:].ravel())*100:.2f}%")
    print(f"Cheat sheet      WAPE: {wape(yte, F['cell_dowhour_mean'][c2:].ravel())*100:.2f}%")
    print(f"YOUR MODEL       WAPE: {wape(yte, p50)*100:.2f}%")
    print("--------------------------------------------------")
    print(f"Pinball q10/q50/q90: {pinball(yte,p10,.1):.3f} / "
          f"{pinball(yte,p50,.5):.3f} / {pinball(yte,p90,.9):.3f}")
    cov = float(np.mean((yte >= p10) & (yte <= p90)))
    print(f"Coverage (q10-q90):  {cov*100:.2f}%  (nominal 80%)")

    # conditional coverage: aggregate coverage can hide bucket-level failure
    width = p90 - p10
    bins = np.digitize(yte, [0.5, 4.5, 19.5])   # 0 / 1-4 / 5-19 / 20+
    print("\n--- Conditional coverage by true-demand bucket ---")
    print(f"{'bucket':>8} | {'n':>9} | {'coverage':>8} | {'mean width':>10}")
    for b, label in enumerate(["=0", "1-4", "5-19", "20+"]):
        m = bins == b
        if m.any():
            print(f"{label:>8} | {m.sum():>9,} | {np.mean((yte[m]>=p10[m])&(yte[m]<=p90[m]))*100:>7.2f}% | "
                  f"{width[m].mean():>10.2f}")

    # reliability (empirical fraction below each predicted quantile)
    print("\n--- Reliability ---")
    for q, p in [(0.1, p10), (0.5, p50), (0.9, p90)]:
        print(f"predicted q{int(q*100)}: empirical fraction below = {np.mean(yte <= p):.3f}")

    # per-cell WAPE (row = slot-major, so cell index = row % n_cells)
    cell_idx = np.tile(np.arange(n_cells), n_test)
    abs_err = np.abs(yte - p50)
    err_sum = np.bincount(cell_idx, weights=abs_err, minlength=n_cells)
    dem_sum = np.bincount(cell_idx, weights=yte, minlength=n_cells)
    wape_cell = pd.DataFrame({"h3_cell": np.array(cells), "abs_err": err_sum,
                              "demand": dem_sum})
    wape_cell["WAPE"] = wape_cell["abs_err"] / wape_cell["demand"]
    print("\nTop 10 H3 cells by WAPE:")
    print(wape_cell.sort_values("WAPE", ascending=False).head(10).to_string(index=False))

    imp = pd.Series(models[0.5].feature_importance("gain"), index=FEATURE_NAMES)
    print("\nTop 10 features (gain):")
    print(imp.sort_values(ascending=False).head(10).to_string())

    # plots (headless-safe)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(width, bins=30)
    ax.set_title("Prediction Interval Widths (q10-q90), holdout")
    fig.savefig(out_dir / "interval_width_hist.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(5, 5))
    qs = [0.1, 0.5, 0.9]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(qs, [np.mean(yte <= p) for p in (p10, p50, p90)], "o-", label="Model")
    ax.set_xlabel("Predicted quantile"); ax.set_ylabel("Empirical fraction below")
    ax.set_title("Reliability diagram (holdout)"); ax.legend()
    fig.savefig(out_dir / "reliability.png", dpi=120, bbox_inches="tight")
    print(f"\n[PLOTS] saved -> {out_dir}")

if __name__ == "__main__":
    main()
