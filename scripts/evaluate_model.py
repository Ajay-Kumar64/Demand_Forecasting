#---------------------------
#USING COLAB FOR EVALUATION
#-------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle  # Use pickle for models saved with pickle.dump
import seaborn as sns
from tqdm.notebook import tqdm  # Import tqdm for progress bar
import time  # Import time for ETA calculation

# ------------------------------
# 1. Load data and model
# ------------------------------
DATA_PATH = "/content/drive/MyDrive/demand_forecasting/h3_features_2025.parquet"
MODEL_PATH = "/content/drive/MyDrive/demand_forecasting/lgbm_quantile_models.pkl"
df = pd.read_parquet(DATA_PATH)
with open(MODEL_PATH, 'rb') as f:
    models_dict = pickle.load(f)  # Rename 'model' to 'models_dict' to reflect its structure

# Drop target for features
X = df.drop(columns=['demand', 'ts_15min',
                     'h3_cell'])  # Ensure 'ts_15min' and 'h3_cell' are dropped for prediction if they were not features during training
y_true = df['demand'].values

# ------------------------------
# 2. Predict quantiles
# ------------------------------
quantiles = [0.1, 0.5, 0.9]
preds = {}

print("\n[PREDICTING] Quantile predictions with ETA...")
for q in quantiles:
    print(f"  Predicting for Quantile alpha={q}")
    quantile_models = models_dict[q]

    q_preds_list = []
    start_time_quantile = time.time()

    for i, model in enumerate(tqdm(quantile_models, desc=f"    Models for q={q}", leave=False)):
        q_preds_list.append(model.predict(X))

    q_preds = np.array(q_preds_list)
    preds[q] = np.mean(q_preds, axis=0)

    end_time_quantile = time.time()
    time_taken_quantile = end_time_quantile - start_time_quantile
    print(f"    Quantile {q} prediction completed in {time_taken_quantile:.2f} seconds.")

print("[PREDICTING] All quantile predictions completed.")


# ------------------------------
# 3. Pinball Loss
# ------------------------------
def pinball_loss(y_true, y_pred, q):
    delta = y_true - y_pred
    return np.mean(np.maximum(q * delta, (q - 1) * delta))


print("\n=== Pinball Loss ===")
for q in quantiles:
    loss = pinball_loss(y_true, preds[q], q)
    print(f"q{int(q * 100)}: {loss:.4f}")

# ------------------------------
# 4. Prediction Interval Coverage
# ------------------------------
# % of true values inside q10-q90 interval
lower = preds[0.1]
upper = preds[0.9]
coverage = np.mean((y_true >= lower) & (y_true <= upper))
print(f"\nPrediction Interval (10-90%) Coverage: {coverage * 100:.2f}%")

# Interval width distribution
plt.figure(figsize=(8, 4))
sns.histplot(upper - lower, bins=30, kde=True)
plt.title("Distribution of Prediction Interval Widths (q10-q90)")
plt.xlabel("Width")
plt.ylabel("Count")
plt.show()


# ------------------------------
# 5. Calibration Check (Reliability)
# ------------------------------
def calibration(y_true, y_pred_quantiles, quantiles):
    actual_frac = []
    for q in quantiles:
        frac = np.mean(y_true <= y_pred_quantiles[q])
        actual_frac.append(frac)
    return actual_frac


actual_frac = calibration(y_true, preds, quantiles)

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
plt.plot(quantiles, actual_frac, marker='o', label='Model')
plt.xlabel("Predicted Quantile")
plt.ylabel("Fraction Below Prediction")
plt.title("Reliability Diagram (Calibration)")
plt.legend()
plt.show()

# ------------------------------
# 6. WAPE (Weighted Absolute Percentage Error)
# ------------------------------
y_pred_median = preds[0.5]
wape = np.sum(np.abs(y_true - y_pred_median)) / np.sum(y_true)
print(f"\nOverall WAPE: {wape * 100:.2f}%")

# WAPE per H3 cell
wape_cell = df.copy()
wape_cell['pred'] = y_pred_median
wape_cell['abs_err'] = np.abs(wape_cell['demand'] - wape_cell['pred'])
wape_per_cell = wape_cell.groupby('h3_cell').agg({'abs_err': 'sum', 'demand': 'sum'})
wape_per_cell['WAPE'] = wape_per_cell['abs_err'] / wape_per_cell['demand']

# Show top 10 worst cells
print("\nTop 10 H3 cells by WAPE:")
print(wape_per_cell.sort_values('WAPE', ascending=False).head(10))

# ------------------------------
# 7. Time Series Plot Example
# ------------------------------
# Plot actual vs predicted for a random H3 cell
example_cell = df['h3_cell'].iloc[0]
df_cell = df[df['h3_cell'] == example_cell].copy()

# Extract features for the example cell
X_example_cell = df_cell.drop(columns=['demand', 'ts_15min', 'h3_cell'])

# Predict median for the example cell using the averaged 0.5 quantile predictions
df_cell['pred_median'] = preds[0.5][
    df['h3_cell'] == example_cell]  # Select predictions corresponding to the example cell
df_cell = df_cell.sort_values('ts_15min')

plt.figure(figsize=(12, 4))
plt.plot(df_cell['ts_15min'], df_cell['demand'], label='Actual', marker='o')
plt.plot(df_cell['ts_15min'], df_cell['pred_median'], label='Predicted (Median)', marker='x')
plt.title(f"H3 Cell {example_cell} Actual vs Predicted")
plt.xlabel("Timestamp")
plt.ylabel("Demand")
plt.legend()
plt.show()