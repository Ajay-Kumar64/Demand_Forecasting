"""
Train quantile LightGBM models on a zero-filled H3 cell x 15-min demand grid.

Replaces the original model_training.py, whose TimeSeriesSplit ran on
cell-sorted rows and therefore validated on unseen CELLS, not unseen TIME.

Pipeline:
  1. Load sparse (h3_cell, ts_15min, demand) aggregates, drop invalid timestamps.
  2. Build a dense (slot x cell) grid, zero-filling empty cell-slots — the
     old pipeline never produced zero-demand rows, biasing the model upward
     and corrupting lag semantics ("672 existing rows ago" != "7 days ago").
  3. Engineer features as pure shifts of the grid (strictly backward-looking).
  4. Split strictly chronologically: 80% train / 10% calibration / 10% test.
  5. Train ONE LightGBM quantile model per alpha (no fold averaging — the mean
     of five q90 predictions is not a q90).
  6. Save models + train-derived lookup tables (cell_mean, dow-hour profile)
     so evaluation and batch inference consume identical features.

Run:  python -m src.modeling.train \
        --demand-parquet data/feature_store/h3_demand_2025.parquet \
        --out-dir artifacts/models
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# ----------------------------- constants -----------------------------
FREQ_MIN = 15
SLOTS_PER_HOUR = 60 // FREQ_MIN                       # 4
LAG_HISTORY_SLOTS = 168 * SLOTS_PER_HOUR              # 672 (7-day warm-up)
SEED = 42
ALPHAS = (0.1, 0.5, 0.9)
K_SMOOTH = 50.0                                       # profile shrinkage strength

MAT_KEYS = [
    "demand_t-15m", "demand_t-1h", "demand_t-4h", "demand_t-168h",
    "demand_roll_mean_3h", "demand_roll_var_3h",
    "demand_roll_mean_6h", "demand_roll_var_6h",
    "cell_dowhour_mean",
]
VEC_KEYS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday"]
FEATURE_NAMES = MAT_KEYS + VEC_KEYS + ["cell_mean", "cell_id"]
CATEGORICAL = ["cell_id"]

# ----------------------------- metrics -----------------------------
def wape(y_true, y_pred) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true))

def pinball(y_true, y_pred, q: float) -> float:
    d = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))

# ----------------------------- data -----------------------------
def load_sparse_demand(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, columns=["h3_cell", "ts_15min", "demand"])
    df["ts_15min"] = pd.to_datetime(df["ts_15min"]).dt.floor(f"{FREQ_MIN}min")

    # TLC vendor files occasionally contain years-old pickup timestamps
    # (observed: 2007-12 rows in the 2025 files). Keep only the dominant year.
    year = int(df["ts_15min"].dt.year.mode()[0])
    bad = df["ts_15min"].dt.year != year
    if bad.any():
        print(f"[DATA-QUALITY] dropping {bad.sum()} rows outside {year} "
              f"(vendor timestamp errors)")
        df = df[~bad]

    df = df.groupby(["h3_cell", "ts_15min"], as_index=False)["demand"].sum()
    print(f"[LOAD] {len(df):,} sparse rows for {year}")
    return df

# ----------------------------- grid & features -----------------------------
def build_grid(sparse: pd.DataFrame):
    """Dense demand grid: rows = 15-min slots, cols = H3 cells. Zeros included."""
    cells = np.sort(sparse["h3_cell"].unique())
    cell_to_j = {c: j for j, c in enumerate(cells)}
    t0 = sparse["ts_15min"].min()

    slot_i = ((sparse["ts_15min"] - t0)
              // pd.Timedelta(minutes=FREQ_MIN)).astype(int).values
    col_j = sparse["h3_cell"].map(cell_to_j).values

    n_slots, n_cells = int(slot_i.max()) + 1, len(cells)
    assert n_slots < 40_000, f"Grid too large ({n_slots} slots) — timestamps still dirty?"

    D = np.zeros((n_slots, n_cells), dtype=np.float32)
    np.add.at(D, (slot_i, col_j), sparse["demand"].values.astype(np.float32))
    ts_slots = pd.date_range(t0, periods=n_slots, freq=f"{FREQ_MIN}min")
    print(f"[GRID] {n_slots} slots x {n_cells} cells")
    return D, cells, ts_slots

def shift_slots(M: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(M, np.nan)
    out[k:] = M[:-k]
    return out

def slot_bucket(ts_slots: pd.DatetimeIndex) -> np.ndarray:
    """day-of-week x hour-of-day bucket index in [0, 168)."""
    return (ts_slots.dayofweek.values * 24 + ts_slots.hour.values).astype(np.int32)

def build_feature_layers(D: np.ndarray, ts_slots: pd.DatetimeIndex):
    """All features are backward-looking shifts of the demand grid."""
    F: dict[str, np.ndarray] = {}
    for name, hours in [("demand_t-15m", 0.25), ("demand_t-1h", 1),
                        ("demand_t-4h", 4), ("demand_t-168h", 168)]:
        F[name] = shift_slots(D, int(round(hours * SLOTS_PER_HOUR)))

    Ddf = pd.DataFrame(D)
    for win_slots, nm in [(12, "3h"), (24, "6h")]:
        F[f"demand_roll_mean_{nm}"] = (Ddf.rolling(win_slots).mean()
                                       .shift(1).to_numpy(dtype=np.float32))
        F[f"demand_roll_var_{nm}"] = (Ddf.rolling(win_slots).var()
                                      .shift(1).to_numpy(dtype=np.float32))
    del Ddf

    h, dow = ts_slots.hour.values, ts_slots.dayofweek.values
    slotvec = {
        "hour_sin": np.sin(2 * np.pi * h / 24).astype(np.float32),
        "hour_cos": np.cos(2 * np.pi * h / 24).astype(np.float32),
        "dow_sin":  np.sin(2 * np.pi * dow / 7).astype(np.float32),
        "dow_cos":  np.cos(2 * np.pi * dow / 7).astype(np.float32),
        "is_weekend": (dow >= 5).astype(np.float32),
    }
    hol = USFederalHolidayCalendar().holidays(start=ts_slots.min(), end=ts_slots.max())
    slotvec["is_holiday"] = ts_slots.normalize().isin(hol).astype(np.float32)
    return F, slotvec

def chronological_splits(n_slots: int):
    start = LAG_HISTORY_SLOTS                    # first week reserved for lags
    n_use = n_slots - start
    c1 = start + int(0.80 * n_use)               # train end / calib start
    c2 = start + int(0.90 * n_use)               # calib end  / test start
    return start, c1, c2

def build_train_lookups(D: np.ndarray, ts_slots: pd.DatetimeIndex,
                        start: int, c1: int, k_smooth: float = K_SMOOTH):
    """cell_mean + smoothed 168-bucket dow-hour profile, TRAIN WINDOW ONLY."""
    Dtr = D[start:c1]
    prior = float(Dtr.mean())
    cell_mean = Dtr.mean(axis=0).astype(np.float32)

    bucket = slot_bucket(ts_slots)
    n_cells = D.shape[1]
    Psum = np.zeros((168, n_cells))
    CNT = np.zeros((168, n_cells), dtype=np.float32)
    bi = np.repeat(bucket[start:c1], n_cells)
    cj = np.tile(np.arange(n_cells, dtype=np.int32), c1 - start)
    np.add.at(Psum, (bi, cj), Dtr.astype(np.float64).ravel())
    np.add.at(CNT, (bi, cj), 1.0)
    profile = ((Psum + prior * k_smooth) / (CNT + k_smooth)).astype(np.float32)

    lookups = {"prior": prior, "cell_mean": cell_mean, "profile_dowhour": profile}
    print("[LOOKUPS] cell_mean + 168x%d dow-hour profile built (train-only)" % n_cells)
    return lookups

# ----------------------------- matrices -----------------------------
def build_matrix(F, slotvec, cell_mean, a: int, b: int, n_cells: int) -> np.ndarray:
    cols = [F[k][a:b].ravel() for k in MAT_KEYS]
    cols += [np.repeat(slotvec[k][a:b], n_cells) for k in VEC_KEYS]
    cols.append(np.tile(cell_mean, b - a))
    cols.append(np.tile(np.arange(n_cells, dtype=np.float32), b - a))
    return np.stack(cols, axis=1).astype(np.float32, copy=False)

def as_lgb_frame(X: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=FEATURE_NAMES, copy=False)
    df["cell_id"] = df["cell_id"].astype(np.int32)     # BUG-9: integer categorical codes
    return df
  
def predict_quantiles(models, X):
    """Predict q10/q50/q90. Per-row rearrangement fixes quantile crossing
    (independently trained quantile models can invert), then clips at zero.
    Order: sort -> clip. Guarantees 0 <= q10 <= q50 <= q90, conf_width >= 0."""
    if isinstance(X, np.ndarray):
        X = as_lgb_frame(X)
    P = np.stack([models[a].predict(X) for a in ALPHAS], axis=1)
    P = np.sort(P, axis=1)          # BUG-6
    P = np.maximum(P, 0.0)          # DEC-A (after sort)
    return P[:, 0], P[:, 1], P[:, 2]
  
# ----------------------------- training -----------------------------
def train_models(Xtr, ytr, Xca, yca) -> dict:
    dtr = lgb.Dataset(Xtr, ytr, feature_name=FEATURE_NAMES,
                      categorical_feature=CATEGORICAL, free_raw_data=True)
    dca = lgb.Dataset(Xca, yca, reference=dtr)

    models = {}
    for alpha in ALPHAS:
        params = {
            "objective": "quantile", "alpha": alpha,
            "learning_rate": 0.05, "num_leaves": 31,
            "seed": SEED, "deterministic": True,       # BUG-8: reproducibility
            "verbose": -1,
        }
        m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dca],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        models[alpha] = m
        print(f"[TRAIN] alpha={alpha} best_iter={m.best_iteration}")
    return models

# ----------------------------- artifacts -----------------------------
def save_artifacts(out_dir: Path, models, lookups, cells, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "lgbm_quantile_models.pkl", "wb") as f:
        pickle.dump({a: models[a] for a in sorted(models)}, f)   # BUG-6: sorted, one model per quantile
    np.save(out_dir / "cell_mean.npy", lookups["cell_mean"])
    np.save(out_dir / "profile_dowhour.npy", lookups["profile_dowhour"])
    (out_dir / "cells.json").write_text(json.dumps(list(cells)))
    (out_dir / "feature_names.json").write_text(json.dumps(FEATURE_NAMES))
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[SAVED] artifacts -> {out_dir}")

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demand-parquet", default="data/feature_store/h3_demand_2025.parquet")
    ap.add_argument("--out-dir", default="artifacts/models")
    args = ap.parse_args()

    sparse = load_sparse_demand(Path(args.demand_parquet))
    D, cells, ts_slots = build_grid(sparse)
    del sparse
    F, slotvec = build_feature_layers(D, ts_slots)

    start, c1, c2 = chronological_splits(D.shape[0])
    print(f"[SPLIT] train {ts_slots[start]} -> {ts_slots[c1-1]} | "
          f"calib -> {ts_slots[c2-1]} | test -> {ts_slots[-1]}")

    lookups = build_train_lookups(D, ts_slots, start, c1)
    F["cell_dowhour_mean"] = lookups["profile_dowhour"][slot_bucket(ts_slots), :]

    n_cells = D.shape[0:1][0] if False else D.shape[1]
    Xtr = build_matrix(F, slotvec, lookups["cell_mean"], start, c1, n_cells)
    ytr = D[start:c1].ravel().astype(np.float32)
    Xca = build_matrix(F, slotvec, lookups["cell_mean"], c1, c2, n_cells)
    yca = D[c1:c2].ravel().astype(np.float32)
    Xte = build_matrix(F, slotvec, lookups["cell_mean"], c2, D.shape[0], n_cells)
    yte = D[c2:].ravel().astype(np.float32)
    print(f"[DATA] train={len(ytr):,} calib={len(yca):,} test={len(yte):,}")

    models = train_models(as_lgb_frame(Xtr), ytr, as_lgb_frame(Xca), yca)

    # quick self-check on the holdout so one command reproduces the headline
    p10, p50, p90 = predict_quantiles(models, Xte)
    print("\n===== HOLDOUT SELF-CHECK =====")
    print(f"Test WAPE (q50):      {wape(yte, p50) * 100:.2f}%")
    print(f"Test Pinball q50:     {pinball(yte, p50, 0.5):.4f}")
    print(f"Test Coverage 10-90:  {np.mean((yte >= p10) & (yte <= p90)) * 100:.2f}%")

    meta = {
        "t0": str(ts_slots[0]), "n_slots": int(D.shape[0]), "n_cells": int(n_cells),
        "start": start, "c1": c1, "c2": c2, "seed": SEED,
        "alphas": list(ALPHAS), "k_smooth": K_SMOOTH,
        "best_iterations": {str(a): models[a].best_iteration for a in models},
        "test_wape_q50": wape(yte, p50),
    }
    save_artifacts(Path(args.out_dir), models, lookups, cells, meta)

if __name__ == "__main__":
    main()
