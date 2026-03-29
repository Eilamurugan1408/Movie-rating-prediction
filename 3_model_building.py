# 3_model_building.py
# Goal: Train multiple models and compare them.
# We try Linear Regression, Random Forest, Gradient Boosting, and KNN.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle, os

# ── Load Split Data ────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()  # squeeze turns DataFrame → Series
y_test  = pd.read_csv("data/y_test.csv").squeeze()

# ── Define Models to Compare ──────────────────────────────────────────────────
# Each entry is (name, model_object)
models = {
    "Linear Regression":    LinearRegression(),
    "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "K-Nearest Neighbors":  KNeighborsRegressor(n_neighbors=10),
}

os.makedirs("models", exist_ok=True)
results = {}

# ── Train, Predict, Evaluate Each Model ───────────────────────────────────────
for name, model in models.items():
    print(f"\nTraining: {name} ...")
    
    # Fit on training data
    model.fit(X_train, y_train)
    
    # Predict on the held-out test set
    y_pred = model.predict(X_test)
    
    # Clip predictions to valid rating range [1, 5]
    y_pred = np.clip(y_pred, 1.0, 5.0)
    
    # Calculate error metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    
    results[name] = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "model": model}
    print(f"  RMSE: {rmse:.4f}  |  MAE: {mae:.4f}")
    
    # Save every model (we'll pick the best one next)
    model_filename = f"models/{name.replace(' ', '_').lower()}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

# ── Summary Table ──────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
summary = pd.DataFrame({k: v for k, v in results.items() if k != "model"}).T
# Reconstruct properly
summary_df = pd.DataFrame([
    {"Model": k, "RMSE": v["RMSE"], "MAE": v["MAE"]}
    for k, v in results.items()
]).set_index("Model").sort_values("RMSE")
print(summary_df)

# Save summary for reference
summary_df.to_csv("data/model_results.csv")