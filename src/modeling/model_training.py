
#---------------------------
#USING COLAB FOR FASTER TRAINING
# -----------------------------
# STEP 0: Install dependencies
# -----------------------------
# !pip install --quiet pandas numpy lightgbm scikit-learn

# -----------------------------
# STEP 1: Mount Google Drive
# -----------------------------
from google.colab import drive
drive.mount('/content/drive')

# -----------------------------
# STEP 2: Paths
# -----------------------------
INPUT_PATH = "/content/drive/MyDrive/demand_forecasting/h3_features_2025.parquet"
OUTPUT_PATH = "/content/drive/MyDrive/demand_forecasting/lgbm_quantile_models.pkl"

# -----------------------------
# STEP 3: Imports
# -----------------------------
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import pickle

# -----------------------------
# STEP 4: Metrics
# -----------------------------
def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

def pinball_loss(y_true, y_pred, alpha=0.5):
    delta = y_true - y_pred
    return np.mean(np.maximum(alpha * delta, (alpha - 1) * delta))

# -----------------------------
# STEP 5: Load data
# -----------------------------
print(f"[LOAD] {INPUT_PATH}")
df = pd.read_parquet(INPUT_PATH)

feature_cols = [c for c in df.columns if c not in ["ts_15min", "demand", "h3_cell"]]
X = df[feature_cols]
y = df["demand"]

print(f"[INFO] {len(X)} rows, {len(feature_cols)} features")

# -----------------------------
# STEP 6: Training function
# -----------------------------
def train_quantile_lgb(X, y, alphas=[0.1,0.5,0.9], n_splits=5, max_iter=500):
    models = {}
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for alpha in alphas:
        print(f"\n[TRAIN] Quantile alpha={alpha}")
        models[alpha] = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"  Fold {fold+1}/{n_splits}")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

            params = {
                "objective": "quantile",
                "alpha": alpha,
                "boosting_type": "gbdt",
                "metric": "quantile",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "verbose": -1,
            }

            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)]
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=max_iter,
                valid_sets=[lgb_train, lgb_val],
                callbacks=callbacks
            )

            models[alpha].append(model)

            # Evaluate
            y_val_pred = model.predict(X_val)
            print(f"    WAPE: {wape(y_val, y_val_pred):.4f}")
            print(f"    Pinball Loss: {pinball_loss(y_val, y_val_pred, alpha):.4f}")

    return models

# -----------------------------
# STEP 7: Train models
# -----------------------------
models = train_quantile_lgb(X, y, alphas=[0.1,0.5,0.9], n_splits=5, max_iter=500)

# -----------------------------
# STEP 8: Save models
# -----------------------------
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(models, f)

print(f"[SAVED] Models to {OUTPUT_PATH}")

# -----------------------------
# STEP 9: Download models (optional)
# -----------------------------
from google.colab import files
files.download(OUTPUT_PATH)