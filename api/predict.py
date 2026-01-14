import numpy as np
import pandas as pd
from api.model_loader import MODELS

def predict_quantiles(df: pd.DataFrame):
    """
    df: DataFrame including identifiers and numeric features
    Returns: DataFrame with identifiers + quantile predictions
    """
    # Columns to use for prediction (numeric only)
    feature_cols = [c for c in df.columns if c not in ["h3_cell", "ts_15min"]]
    X = df[feature_cols].astype(float)  # ensure numeric

    preds = {}
    for alpha, model_list in MODELS.items():
        model = model_list[-1]  # use last fold model
        preds[alpha] = model.predict(X)

    df_out = df[["h3_cell", "ts_15min"]].copy()
    df_out["q10"] = np.maximum(preds[0.1], 0)
    df_out["q50"] = np.maximum(preds[0.5], 0)
    df_out["q90"] = np.maximum(preds[0.9], 0)
    df_out["conf_width"] = df_out["q90"] - df_out["q10"]

    return df_out
