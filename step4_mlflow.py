"""
Step 4: MLflow Integration
Tracks experiments, logs metrics, registers best models.
Run AFTER step3_train_models.py
Usage: python step4_mlflow.py
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             mean_squared_error, mean_absolute_error, r2_score)

# ─── LOAD DATA ────────────────────────────────────────────────
print("Loading processed data...")
df = pd.read_csv('data/processed_data.csv')

FEATURES = [
    'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Floor_No', 'Total_Floors',
    'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals',
    'Price_per_SqFt_Calc', 'School_Density_Score', 'Hospital_Density_Score',
    'Infrastructure_Score', 'Is_Ready_to_Move', 'Transport_Score',
    'Parking_Numeric', 'Security_Score', 'Amenity_Score',
    'Price_vs_CityMedian',
    'Property_Type_Enc', 'Furnished_Status_Enc', 'Facing_Enc',
    'Owner_Type_Enc', 'Availability_Status_Enc', 'City_Enc', 'State_Enc'
]

X = df[FEATURES]
y_class = df['Good_Investment']
y_reg   = df['Future_Price_5Y']

X_train, X_test, yc_train, yc_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class)
_, _, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── MLFLOW SETUP ─────────────────────────────────────────────
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("RealEstate_Investment_Advisor")

print("Starting MLflow experiment tracking...")

# ══════════════════════════════════════════════════════════════
# CLASSIFICATION EXPERIMENTS
# ══════════════════════════════════════════════════════════════

clf_configs = [
    {
        "name": "Logistic_Regression_Clf",
        "model": LogisticRegression(max_iter=500, C=1.0, random_state=42),
        "params": {"max_iter": 500, "C": 1.0},
        "scaled": True
    },
    {
        "name": "RandomForest_Clf_100",
        "model": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "params": {"n_estimators": 100, "max_depth": 12},
        "scaled": False
    },
    {
        "name": "RandomForest_Clf_200",
        "model": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        "params": {"n_estimators": 200, "max_depth": 15},
        "scaled": False
    },
]

best_clf_run_id = None
best_clf_f1 = 0.0

for cfg in clf_configs:
    with mlflow.start_run(run_name=cfg["name"]) as run:
        mlflow.set_tag("task", "classification")
        mlflow.set_tag("target", "Good_Investment")
        mlflow.log_params(cfg["params"])
        mlflow.log_param("features_count", len(FEATURES))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size",  len(X_test))

        Xtr = X_train_sc if cfg["scaled"] else X_train
        Xte = X_test_sc  if cfg["scaled"] else X_test

        cfg["model"].fit(Xtr, yc_train)
        y_pred = cfg["model"].predict(Xte)

        acc = accuracy_score(yc_test, y_pred)
        f1  = f1_score(yc_test, y_pred)
        cm  = confusion_matrix(yc_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("precision", tp / (tp + fp) if (tp+fp) else 0)
        mlflow.log_metric("recall",    tp / (tp + fn) if (tp+fn) else 0)
        mlflow.log_metric("true_positives",  int(tp))
        mlflow.log_metric("false_positives", int(fp))
        mlflow.log_metric("true_negatives",  int(tn))
        mlflow.log_metric("false_negatives", int(fn))

        mlflow.sklearn.log_model(cfg["model"], artifact_path="model",
                                  registered_model_name=f"clf_{cfg['name']}")

        print(f"  [{cfg['name']}] acc={acc:.4f} f1={f1:.4f} run_id={run.info.run_id[:8]}")

        if f1 > best_clf_f1:
            best_clf_f1 = f1
            best_clf_run_id = run.info.run_id

print(f"\n✅ Best Classifier run_id: {best_clf_run_id[:8]} (F1={best_clf_f1:.4f})")

# ══════════════════════════════════════════════════════════════
# REGRESSION EXPERIMENTS
# ══════════════════════════════════════════════════════════════

reg_configs = [
    {
        "name": "Linear_Regression_Reg",
        "model": LinearRegression(),
        "params": {"fit_intercept": True},
        "scaled": True
    },
    {
        "name": "RandomForest_Reg_100",
        "model": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "params": {"n_estimators": 100, "max_depth": 12},
        "scaled": False
    },
    {
        "name": "RandomForest_Reg_200",
        "model": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        "params": {"n_estimators": 200, "max_depth": 15},
        "scaled": False
    },
]

best_reg_run_id = None
best_reg_rmse = float('inf')

for cfg in reg_configs:
    with mlflow.start_run(run_name=cfg["name"]) as run:
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("target", "Future_Price_5Y")
        mlflow.log_params(cfg["params"])
        mlflow.log_param("features_count", len(FEATURES))

        Xtr = X_train_sc if cfg["scaled"] else X_train
        Xte = X_test_sc  if cfg["scaled"] else X_test

        cfg["model"].fit(Xtr, yr_train)
        y_pred = cfg["model"].predict(Xte)

        rmse = np.sqrt(mean_squared_error(yr_test, y_pred))
        mae  = mean_absolute_error(yr_test, y_pred)
        r2   = r2_score(yr_test, y_pred)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE",  mae)
        mlflow.log_metric("R2",   r2)

        mlflow.sklearn.log_model(cfg["model"], artifact_path="model",
                                  registered_model_name=f"reg_{cfg['name']}")

        print(f"  [{cfg['name']}] RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} run_id={run.info.run_id[:8]}")

        if rmse < best_reg_rmse:
            best_reg_rmse = rmse
            best_reg_run_id = run.info.run_id

print(f"\n✅ Best Regressor run_id: {best_reg_run_id[:8]} (RMSE={best_reg_rmse:.4f})")

print("\n" + "=" * 60)
print("MLflow tracking complete.")
print("View UI: mlflow ui  →  http://localhost:5000")
print("=" * 60)
