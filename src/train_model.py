# src/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

# Load the dataset
data = pd.read_csv("../data/DrivingDataset.csv")

# Prepare features and label
X = data.drop(columns=["labels", "trajectory_id", "start_time", "end_time"])
y = data["labels"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Report:\n", classification_report(y_test, y_pred_rf))
joblib.dump(rf, "../models/random_forest_model.pkl")

# MLP Classifier
print("Training MLP...")
mlp = MLPClassifier(max_iter=1000)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print("\nMLP Classifier Report:\n", classification_report(y_test, y_pred_mlp))
joblib.dump(mlp, "../models/mlp_model.pkl")
