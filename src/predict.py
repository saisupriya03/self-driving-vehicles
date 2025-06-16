# src/predict.py

import pandas as pd
import joblib

# Load test data (without labels)
test_data = pd.read_csv("../data/test_data.csv")
X_test = test_data.drop(columns=["trajectory_id", "start_time", "end_time"])

# Load trained models
rf_model = joblib.load("../models/random_forest_model.pkl")
mlp_model = joblib.load("../models/mlp_model.pkl")

# Predict using Random Forest
rf_preds = rf_model.predict(X_test)
print("\nRandom Forest Predictions:")
print(rf_preds)

# Predict using MLP
mlp_preds = mlp_model.predict(X_test)
print("\nMLP Predictions:")
print(mlp_preds)
