import pickle

MODEL_PATH = "artifacts/models/lgbm_quantile_models.pkl"

def load_models(path=MODEL_PATH):
    with open(path, "rb") as f:
        models = pickle.load(f)
    print("[INFO] Models loaded")
    return models

# Load once at startup
MODELS = load_models()
