Demand Forecasting System

Production-Grade Demand Forecasting Platform for high-throughput, accurate predictions across multiple time horizons. 
Built for scalable deployment in real-world business environments, enabling data-driven inventory and resource planning decisions.

Key Features

Quantile Forecasting – Provides probabilistic forecasts to capture uncertainty, including multiple quantiles (e.g., 0.1, 0.5, 0.9).

High-Throughput Pipeline – Optimized batch and streaming prediction pipelines capable of handling millions of records efficiently.

Feature Engineering – Includes lag features, rolling statistics, calendar features, and external covariates (e.g., holidays, promotions).

Modeling – Uses state-of-the-art time series models (Prophet, XGBoost, LightGBM, NeuralProphet, or custom ensemble models).

Evaluation Metrics – Supports MAPE, RMSE, Quantile Loss, and coverage metrics for probabilistic forecasts.

Extensible Architecture – Modular design for adding new models, datasets, or feature sets without affecting the pipeline.

Production Deployment Ready – Dockerized pipelines, reproducible scripts, and asynchronous inference for low-latency predictions.

Tech Stack

Python, Pandas, NumPy, Scikit-learn

LightGBM 

Apache Airflow / Prefect for scheduling

PyTorch / TensorFlow (optional neural models)

FastAPI / Flask for serving predictions

Docker for containerization

Usage

Install dependencies

pip install -r requirements.txt


Preprocess data

python scripts/preprocess.py --input data/raw.csv --output data/processed.csv


Train models

python scripts/train.py --config config/train_config.yaml


Run predictions

python scripts/predict.py --model models/latest_model.pkl --output predictions.csv


Serve via API (optional)

uvicorn inference.api:app --host 0.0.0.0 --port 8000
