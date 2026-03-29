# 5_save_model.py
# Goal: Find the best model and save it as 'best_model.pkl' for the Streamlit app.

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

model_files = {
    "Linear Regression":   "models/linear_regression.pkl",
    "Random Forest":       "models/random_forest.pkl",
    "Gradient Boosting":   "models/gradient_boosting.pkl",
    "K-Nearest Neighbors": "models/k-nearest_neighbors.pkl",
}

best_name, best_rmse, best_model = None, float("inf"), None

for name, path in model_files.items():
    with open(path, "rb") as f:
        model = pickle.load(f)
    y_pred = np.clip(model.predict(X_test), 1.0, 5.0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    if rmse < best_rmse:
        best_rmse, best_name, best_model = rmse, name, model

print(f"Best model selected: {best_name}  (RMSE = {best_rmse:.4f})")

# Save the winning model
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Saved as models/best_model.pkl")