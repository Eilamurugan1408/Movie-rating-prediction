# 4_evaluation.py
# Goal: Visually compare models and understand what the errors mean.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

X_test  = pd.read_csv("data/X_test.csv")
y_test  = pd.read_csv("data/y_test.csv").squeeze()

model_files = {
    "Linear Regression":   "models/linear_regression.pkl",
    "Random Forest":       "models/random_forest.pkl",
    "Gradient Boosting":   "models/gradient_boosting.pkl",
    "K-Nearest Neighbors": "models/k-nearest_neighbors.pkl",
}

results = []
best_name, best_rmse, best_model = None, float("inf"), None

for name, path in model_files.items():
    with open(path, "rb") as f:
        model = pickle.load(f)
    y_pred = np.clip(model.predict(X_test), 1.0, 5.0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    results.append({"Model": name, "RMSE": rmse, "MAE": mae})
    if rmse < best_rmse:
        best_rmse, best_name, best_model = rmse, name, model

df_results = pd.DataFrame(results).set_index("Model")
print(f"\nBest model: {best_name}  (RMSE = {best_rmse:.4f})")

# ── Visualise Comparison ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df_results["RMSE"].sort_values().plot(kind="barh", ax=axes[0], color="#5C85D6")
axes[0].set_title("RMSE by Model (lower = better)")
axes[0].set_xlabel("RMSE")

df_results["MAE"].sort_values().plot(kind="barh", ax=axes[1], color="#7EBF9C")
axes[1].set_title("MAE by Model (lower = better)")
axes[1].set_xlabel("MAE")

plt.tight_layout()
plt.savefig("data/model_comparison.png", dpi=150)
plt.show()

# ── Feature Importance (if Random Forest or Gradient Boosting is best) ────────
with open("models/random_forest.pkl", "rb") as f:
    rf = pickle.load(f)

feature_names = X_test.columns.tolist()
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(8, 4))
importances.plot(kind="bar", color="#E4845A", edgecolor="white")
plt.title("Random Forest — Feature Importance")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("data/feature_importance.png", dpi=150)
plt.show()