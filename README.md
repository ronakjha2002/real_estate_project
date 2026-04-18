# 🏠 Real Estate Investment Advisor
### Predicting Property Profitability & Future Value

---

## 📌 Project Overview

A full machine learning application to assist investors in making real estate decisions using India housing prices data (250,000 properties).

| Task | Model | Performance |
|---|---|---|
| Classification (Good Investment?) | Random Forest | Accuracy: 100%, F1: 1.000 |
| Regression (Price after 5 years) | Random Forest | RMSE: 4.2 Lakhs, R²: 0.9996 |

---

## 🗂️ Project Structure

```
real_estate_project/
│
├── data/
│   └── processed_data.csv          ← cleaned + feature-engineered dataset
│
├── models/
│   ├── best_classifier.pkl         ← trained RF classifier
│   ├── best_regressor.pkl          ← trained RF regressor
│   ├── scaler.pkl                  ← StandardScaler
│   ├── model_metadata.json         ← feature lists, encodings, city maps
│   ├── experiment_log.json         ← all model metrics
│   ├── clf_feature_importance.json ← top features for classification
│   └── reg_feature_importance.json ← top features for regression
│
├── eda_outputs/                    ← 20 EDA charts (PNG)
│   ├── eda01_price_distribution.png
│   ├── eda02_size_distribution.png
│   └── ... (eda03 to eda20)
│
├── step1_preprocess_eda.py         ← Data preprocessing + EDA summary
├── step2_eda_plots.py              ← Generates all 20 EDA charts
├── step3_train_models.py           ← Model training + experiment logging
├── step4_mlflow.py                 ← Full MLflow integration
├── app.py                          ← Streamlit application
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline in order

```bash
# Step 1: Preprocess data + EDA summary
python step1_preprocess_eda.py

# Step 2: Generate EDA charts
python step2_eda_plots.py

# Step 3: Train models
python step3_train_models.py

# Step 4: MLflow tracking (optional, requires mlflow)
python step4_mlflow.py
mlflow ui   # Open http://localhost:5000

# Step 5: Launch Streamlit app
streamlit run app.py
```

---

## 📊 Dataset

- **Source**: `india_housing_prices.xlsx`
- **Size**: 250,000 rows × 23 features
- **Missing values**: None
- **Duplicates**: None

### Key Features
| Feature | Type | Description |
|---|---|---|
| BHK | Numeric | Bedrooms, Hall, Kitchen |
| Size_in_SqFt | Numeric | Property area |
| Price_in_Lakhs | Numeric | Current price |
| City / State | Categorical | Location |
| Property_Type | Categorical | Apartment / Villa / House |
| Amenities | Categorical | Gym, Pool, Clubhouse |
| Infrastructure_Score | Engineered | Composite score |
| Price_vs_CityMedian | Engineered | Relative affordability |

---

## 🎯 Target Variables

### Classification: `Good_Investment` (0 or 1)
**Rule**: A property is a Good Investment if:
- Price ≤ city median **AND** BHK ≥ 2
- OR: Price ≤ city median **AND** Size ≥ dataset median

**Result**: 37.4% of properties classified as Good Investment

### Regression: `Future_Price_5Y` (₹ Lakhs)
**Formula**: `Future = Current × (1 + growth_rate)^5`

| City Tier | Cities | Growth Rate |
|---|---|---|
| Tier 1 (Premium) | Bangalore, Mumbai, Chennai, Hyderabad... | 10% p.a. |
| Tier 2 (Mid) | Jaipur, Kochi, Ahmedabad... | 8.5% p.a. |
| Tier 3 (Emerging) | Ludhiana, Gaya, Bilaspur... | 7% p.a. |

---

## 🔧 Feature Engineering

| Feature | Formula | Purpose |
|---|---|---|
| Price_per_SqFt_Calc | Price × 1e5 / SqFt | Normalized price |
| School_Density_Score | Schools / 10 | Education proximity |
| Hospital_Density_Score | Hospitals / 10 | Healthcare proximity |
| Infrastructure_Score | Sum of transport + parking + security + amenity + density scores | Overall livability |
| Price_vs_CityMedian | Price / City Median | Affordability benchmark |
| Is_Ready_to_Move | Binary from Availability_Status | Investment readiness |

---

## 🤖 Models

### Classification
| Model | Accuracy | F1-Score | Notes |
|---|---|---|---|
| Logistic Regression | 87.1% | 0.8258 | Baseline |
| **Random Forest** | **100%** | **1.000** | ✅ Selected |

### Regression
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 9.98 L | 7.32 L | 0.9978 |
| **Random Forest** | **4.20 L** | **2.21 L** | **0.9996** |

---

## 🖥️ Streamlit App Pages

1. **🔮 Investment Predictor** — Enter property details → get verdict + 5-year forecast
2. **📊 Market Explorer** — Filter by city/price/BHK/type, view summary stats
3. **📈 EDA Insights** — 20 EDA charts across 4 tabs
4. **🧪 Model Performance** — Experiment log, feature importance, model registry

---

## 📈 EDA Coverage

| # | Question |
|---|---|
| 1-2 | Price and size distributions |
| 3 | Price/sqft by property type |
| 4 | Size vs price relationship |
| 5 | Outlier detection |
| 6-7 | Avg price by state and city |
| 8 | Median property age by locality |
| 9 | BHK distribution across cities |
| 10 | Price trends for top 5 localities |
| 11 | Feature correlation heatmap |
| 12-13 | Schools and hospitals vs price |
| 14 | Price by furnished status |
| 15 | Price/sqft by facing direction |
| 16-17 | Owner type and availability distributions |
| 18 | Parking vs price |
| 19 | Amenities vs price/sqft |
| 20 | Transport accessibility vs investment |

---

## 🧪 MLflow Tracking

Each experiment run logs:
- **Parameters**: n_estimators, max_depth, C, etc.
- **Metrics**: Accuracy, F1, RMSE, MAE, R²
- **Artifacts**: Trained model (pkl)
- **Tags**: task type, target variable

View UI: `mlflow ui` → http://localhost:5000

---

## 📦 Deliverables

- [x] Cleaned & processed dataset (CSV)
- [x] Python scripts for EDA, preprocessing, model training
- [x] 20 EDA charts (PNG)
- [x] MLflow experiment tracking script
- [x] Streamlit application (prediction + insights)
- [x] Project documentation (this README)

---

## 🏷️ Technical Tags

`Python` `Pandas` `Scikit-learn` `Random Forest` `Regression` `Classification`
`Streamlit` `MLflow` `Real Estate Analytics` `Data Visualization` `Feature Engineering`
