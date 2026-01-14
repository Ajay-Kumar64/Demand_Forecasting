from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from typing import List

# ----------------------------
# Model Loader
# ----------------------------
MODEL_PATH = "artifacts/models/lgbm_quantile_models.pkl"

def load_models(path=MODEL_PATH):
    """
    Load quantile LightGBM models saved as:
    {0.1: [fold1_model, fold2_model, ...], 0.5: [...], 0.9: [...]}
    """
    with open(path, "rb") as f:
        models = pickle.load(f)
    print("[INFO] Models loaded")
    return models

MODELS = load_models()

# ----------------------------
# Prediction Logic
# ----------------------------
def predict_quantiles(df: pd.DataFrame):
    """
    df: DataFrame with identifiers + numeric features
    Returns: DataFrame with identifiers + quantile predictions
    """
    # Use training feature names from model
    FEATURE_COLS = MODELS[0.5][-1].feature_name()

    # Add missing columns as 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Select features in correct order
    X = df[FEATURE_COLS].astype(float)

    preds = {}
    for alpha, model_list in MODELS.items():
        model = model_list[-1]  # last fold model
        preds[alpha] = model.predict(X)

    # Build output DataFrame
    df_out = df[["h3_cell", "ts_15min"]].copy()
    df_out["q10"] = np.maximum(preds[0.1], 0)
    df_out["q50"] = np.maximum(preds[0.5], 0)
    df_out["q90"] = np.maximum(preds[0.9], 0)
    df_out["conf_width"] = df_out["q90"] - df_out["q10"]

    return df_out

# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI(title="NYC Taxi Demand Forecast API")

# Request / Response Models
class PredictionRequest(BaseModel):
    features: List[dict]  # each dict = row of features

class PredictionResponse(BaseModel):
    h3_cell: str
    ts_15min: str
    q10: float
    q50: float
    q90: float
    conf_width: float

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/")
def root():
    return {"message": "Taxi Demand Forecast API running"}

@app.post("/predict", response_model=List[PredictionResponse])
def predict(request: PredictionRequest):
    if not request.features:
        return []

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(request.features)

    # Check for mandatory identifier columns
    if "h3_cell" not in df.columns or "ts_15min" not in df.columns:
        return {"error": "Missing 'h3_cell' or 'ts_15min' in features"}

    # Run prediction
    df_preds = predict_quantiles(df)

    return df_preds.to_dict(orient="records")
