"""
Step 3: Model Development + Step 4: Experiment Tracking (MLflow-style)
Real Estate Investment Advisor
"""

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
                             classification_report, mean_squared_error,
                             mean_absolute_error, r2_score)

print("=" * 60)
print("STEP 3: MODEL DEVELOPMENT")
print("=" * 60)

# ─────────────────────────────────────────
# 1. LOAD PROCESSED DATA
# ─────────────────────────────────────────
df = pd.read_csv(r'C:\Users\SURYAPRAKASH JHA\Downloads\real_estate_project\real_estate_project\data\processed_data.csv')
print(f"Loaded processed data: {df.shape}")

# ─────────────────────────────────────────
# 2. FEATURE SELECTION
# ─────────────────────────────────────────
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

TARGET_CLASS = 'Good_Investment'
TARGET_REG   = 'Future_Price_5Y'

X = df[FEATURES]
y_class = df[TARGET_CLASS]
y_reg   = df[TARGET_REG]

print(f"\nFeatures: {len(FEATURES)}")
print(f"Class balance: {y_class.value_counts().to_dict()}")

# ─────────────────────────────────────────
# 3. TRAIN/TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, yc_train, yc_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class)

_, _, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

# ─────────────────────────────────────────
# 4. SCALING
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────
# 5. EXPERIMENT LOG (MLflow-style dict)
# ─────────────────────────────────────────
experiment_log = {"classification": [], "regression": []}

# ─────────────────────────────────────────
# 6. CLASSIFICATION MODELS
# ─────────────────────────────────────────
print("\n" + "-" * 40)
print("CLASSIFICATION: Good Investment")
print("-" * 40)

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12,
                                            random_state=42, n_jobs=-1),
}

best_clf = None
best_clf_score = 0
best_clf_name  = ""

for name, model in clf_models.items():
    print(f"\n  Training {name}...")
    if name == "Logistic Regression":
        model.fit(X_train_sc, yc_train)
        y_pred = model.predict(X_test_sc)
    else:
        model.fit(X_train, yc_train)
        y_pred = model.predict(X_test)

    acc  = accuracy_score(yc_test, y_pred)
    f1   = f1_score(yc_test, y_pred)
    cm   = confusion_matrix(yc_test, y_pred).tolist()

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(yc_test, y_pred)}")

    experiment_log["classification"].append({
        "model": name, "accuracy": round(acc, 4),
        "f1_score": round(f1, 4), "confusion_matrix": cm
    })

    if f1 > best_clf_score:
        best_clf_score = f1
        best_clf = model
        best_clf_name = name

print(f"\n  ✅ Best Classifier: {best_clf_name} (F1={best_clf_score:.4f})")

# Feature importance for RF
if hasattr(best_clf, 'feature_importances_'):
    fi = pd.Series(best_clf.feature_importances_, index=FEATURES)
    fi = fi.sort_values(ascending=False).head(10)
    print("\n  Top 10 Feature Importances (Classification):")
    print(fi.round(4).to_string())
    import os
    os.makedirs('models', exist_ok=True)
    fi.to_json('models/clf_feature_importance.json')

# ─────────────────────────────────────────
# 7. REGRESSION MODELS
# ─────────────────────────────────────────
print("\n" + "-" * 40)
print("REGRESSION: Future Price (5 Years)")
print("-" * 40)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12,
                                                      random_state=42, n_jobs=-1),
}

best_reg = None
best_reg_score = float('inf')
best_reg_name  = ""

# Use same X (includes Price_in_Lakhs which is known at prediction time)
X_train_r, X_test_r, yr_train2, yr_test2 = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)
X_train_r_sc = scaler.transform(X_train_r)
X_test_r_sc  = scaler.transform(X_test_r)
yr_test = yr_test2

for name, model in reg_models.items():
    print(f"\n  Training {name}...")
    if name == "Linear Regression":
        model.fit(X_train_r_sc, yr_train2)
        y_pred = model.predict(X_test_r_sc)
    else:
        model.fit(X_train_r, yr_train2)
        y_pred = model.predict(X_test_r)

    rmse = np.sqrt(mean_squared_error(yr_test, y_pred))
    mae  = mean_absolute_error(yr_test, y_pred)
    r2   = r2_score(yr_test, y_pred)

    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

    experiment_log["regression"].append({
        "model": name, "RMSE": round(rmse, 4),
        "MAE": round(mae, 4), "R2": round(r2, 4)
    })

    if rmse < best_reg_score:
        best_reg_score = rmse
        best_reg = model
        best_reg_name = name

print(f"\n  ✅ Best Regressor: {best_reg_name} (RMSE={best_reg_score:.4f})")

if hasattr(best_reg, 'feature_importances_'):
    fi_r = pd.Series(best_reg.feature_importances_, index=FEATURES)
    fi_r = fi_r.sort_values(ascending=False).head(10)
    print("\n  Top 10 Feature Importances (Regression):")
    print(fi_r.round(4).to_string())
    fi_r.to_json('models/reg_feature_importance.json')
# ─────────────────────────────────────────
# 8. SAVE EXPERIMENT LOG (MLflow substitute)
# ─────────────────────────────────────────
log_path = 'models/experiment_log.json'
with open(log_path, 'w') as f:
    json.dump(experiment_log, f, indent=2)
print(f"\n[4] Experiment log saved → {log_path}")

# ─────────────────────────────────────────
# 9. SAVE BEST MODELS & SCALER
# ─────────────────────────────────────────
with open('models/best_classifier.pkl', 'wb') as f:
    pickle.dump(best_clf, f)
with open('models/best_regressor.pkl', 'wb') as f:
    pickle.dump(best_reg, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature list & encodings metadata
meta = {
    "features": FEATURES,
    "best_classifier": best_clf_name,
    "best_regressor": best_reg_name,
    "clf_f1": best_clf_score,
    "reg_rmse": best_reg_score,
    "cities": sorted(df['City'].unique().tolist()),
    "states": sorted(df['State'].unique().tolist()),
    "city_enc_map": df[['City','City_Enc']].drop_duplicates().set_index('City')['City_Enc'].to_dict(),
    "state_enc_map": df[['State','State_Enc']].drop_duplicates().set_index('State')['State_Enc'].to_dict(),
    "property_type_enc_map": df[['Property_Type','Property_Type_Enc']].drop_duplicates().set_index('Property_Type')['Property_Type_Enc'].to_dict(),
    "furnished_enc_map": df[['Furnished_Status','Furnished_Status_Enc']].drop_duplicates().set_index('Furnished_Status')['Furnished_Status_Enc'].to_dict(),
    "city_medians": df.groupby('City')['Price_in_Lakhs'].median().to_dict(),
    "city_growth_rates": df[['City','Growth_Rate']].drop_duplicates().set_index('City')['Growth_Rate'].to_dict(),
}
with open('models/model_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\n✅ All models and metadata saved to /models/")
print("\n" + "=" * 60)
print("EXPERIMENT LOG SUMMARY")
print("=" * 60)
print(json.dumps(experiment_log, indent=2))
