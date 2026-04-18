"""
Step 1: Data Preprocessing + Step 2: EDA
Real Estate Investment Advisor
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("=" * 60)
print("STEP 1: DATA PREPROCESSING")
print("=" * 60)

df = pd.read_excel(r'C:\Users\SURYAPRAKASH JHA\Downloads\india_housing_prices.xlsx')
print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────
# 2. MISSING VALUES & DUPLICATES
# ─────────────────────────────────────────
print("\n[2.1] Missing values per column:")
print(df.isnull().sum())

dup_count = df.duplicated().sum()
print(f"\n[2.2] Duplicate rows: {dup_count}")
df.drop_duplicates(inplace=True)
print(f"     Shape after dedup: {df.shape}")

# ─────────────────────────────────────────
# 3. OUTLIER DETECTION & CAPPING
# ─────────────────────────────────────────
def cap_outliers(series, low=0.01, high=0.99):
    lo, hi = series.quantile(low), series.quantile(high)
    return series.clip(lo, hi)

for col in ['Price_in_Lakhs', 'Size_in_SqFt', 'Price_per_SqFt']:
    df[col] = cap_outliers(df[col])
    print(f"[3] Capped outliers in {col}")

# ─────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────
print("\n[4] Feature Engineering ...")

# Price per sqft (recalculate after capping)
df['Price_per_SqFt_Calc'] = (df['Price_in_Lakhs'] * 1e5) / df['Size_in_SqFt']

# School density score (normalize 0-1)
df['School_Density_Score'] = (df['Nearby_Schools'] - df['Nearby_Schools'].min()) / \
                              (df['Nearby_Schools'].max() - df['Nearby_Schools'].min())

# Hospital density score
df['Hospital_Density_Score'] = (df['Nearby_Hospitals'] - df['Nearby_Hospitals'].min()) / \
                                (df['Nearby_Hospitals'].max() - df['Nearby_Hospitals'].min())

# City median price
city_median = df.groupby('City')['Price_in_Lakhs'].transform('median')
df['Price_vs_CityMedian'] = df['Price_in_Lakhs'] / city_median

# Is RERA-like (Ready to move = safer investment proxy)
df['Is_Ready_to_Move'] = (df['Availability_Status'] == 'Ready_to_Move').astype(int)

# Infrastructure score (composite)
transport_map = {'High': 3, 'Medium': 2, 'Low': 1}
df['Transport_Score'] = df['Public_Transport_Accessibility'].map(transport_map)

parking_map = {'2': 2, '1': 1, 'None': 0}
df['Parking_Numeric'] = df['Parking_Space'].map(parking_map).fillna(0)

security_map = {'Gated + CCTV + Guard': 3, 'Gated + CCTV': 2, 'Gated': 1, 'None': 0}
df['Security_Score'] = df['Security'].map(security_map).fillna(1)

amenity_map = {'Gym + Pool + Clubhouse': 3, 'Gym + Pool': 2, 'Gym': 1, 'None': 0}
df['Amenity_Score'] = df['Amenities'].map(amenity_map).fillna(1)

df['Infrastructure_Score'] = (df['Transport_Score'] + df['Parking_Numeric'] +
                               df['Security_Score'] + df['Amenity_Score'] +
                               df['School_Density_Score'] + df['Hospital_Density_Score'])

# ─────────────────────────────────────────
# 5. TARGET VARIABLES
# ─────────────────────────────────────────
print("\n[5] Creating target variables ...")

# Regression: Future Price (5 years)
# Use city-based growth rate (tiered by current average price)
city_avg = df.groupby('City')['Price_in_Lakhs'].mean()
city_tier = pd.qcut(city_avg, q=3, labels=['Tier3', 'Tier2', 'Tier1'])
growth_rate_map = {'Tier1': 0.10, 'Tier2': 0.085, 'Tier3': 0.07}
df['City_Tier'] = df['City'].map(city_tier).astype(str)
df['Growth_Rate'] = df['City_Tier'].map(growth_rate_map).fillna(0.085)
df['Future_Price_5Y'] = df['Price_in_Lakhs'] * (1 + df['Growth_Rate']) ** 5

# Classification: Good Investment
# Multi-factor: below city median price, decent size, good infra
price_ok = df['Price_vs_CityMedian'] <= 1.0
sqft_ok  = df['Size_in_SqFt'] >= df['Size_in_SqFt'].median()
infra_ok = df['Infrastructure_Score'] >= df['Infrastructure_Score'].quantile(0.4)
bhk_ok   = df['BHK'] >= 2
df['Good_Investment'] = ((price_ok & infra_ok & bhk_ok) | (price_ok & sqft_ok)).astype(int)

print(f"Good Investment distribution:\n{df['Good_Investment'].value_counts()}")
print(f"Good Investment %: {df['Good_Investment'].mean()*100:.1f}%")

# ─────────────────────────────────────────
# 6. ENCODE CATEGORICALS
# ─────────────────────────────────────────
print("\n[6] Encoding categorical features ...")

label_cols = ['Property_Type', 'Furnished_Status', 'Facing',
              'Owner_Type', 'Security', 'Amenities',
              'Public_Transport_Accessibility', 'Parking_Space',
              'Availability_Status', 'City_Tier']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in label_cols:
    df[col + '_Enc'] = le.fit_transform(df[col].astype(str))

# City and State label encoding
df['City_Enc'] = le.fit_transform(df['City'].astype(str))
df['State_Enc'] = le.fit_transform(df['State'].astype(str))

# ─────────────────────────────────────────
# 7. SAVE PROCESSED DATA
# ─────────────────────────────────────────
import os
os.makedirs('data', exist_ok=True)   # ← add this line
out_path = 'data/processed_data.csv'
df.to_csv(out_path, index=False)
print(f"\n[7] Processed data saved → {out_path}")
print(f"    Final shape: {df.shape}")

# ─────────────────────────────────────────
# STEP 2: EDA SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS SUMMARY")
print("=" * 60)

# EDA 1: Price distribution
print("\n[EDA 1] Price_in_Lakhs stats:")
print(df['Price_in_Lakhs'].describe().round(2))

# EDA 2: Size distribution
print("\n[EDA 2] Size_in_SqFt stats:")
print(df['Size_in_SqFt'].describe().round(2))

# EDA 3: Price per SqFt by property type
print("\n[EDA 3] Price_per_SqFt by Property_Type:")
print(df.groupby('Property_Type')['Price_per_SqFt'].mean().round(4).sort_values(ascending=False))

# EDA 4: Correlation
num_cols = ['Price_in_Lakhs', 'Size_in_SqFt', 'BHK', 'Age_of_Property',
            'Nearby_Schools', 'Nearby_Hospitals', 'Infrastructure_Score',
            'Future_Price_5Y', 'Good_Investment']
print("\n[EDA 11] Correlations with Price_in_Lakhs:")
corr = df[num_cols].corr()['Price_in_Lakhs'].sort_values(ascending=False)
print(corr.round(3))

# EDA 6: Avg price per sqft by state
print("\n[EDA 6] Top 10 States by avg Price_in_Lakhs:")
print(df.groupby('State')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(10).round(2))

# EDA 7: Average price by city (top 10)
print("\n[EDA 7] Top 10 Cities by avg Price_in_Lakhs:")
print(df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(10).round(2))

# EDA 9: BHK distribution across top cities
print("\n[EDA 9] BHK distribution (overall):")
print(df['BHK'].value_counts().sort_index())

# EDA 14: Price by furnished status
print("\n[EDA 14] Avg Price by Furnished Status:")
print(df.groupby('Furnished_Status')['Price_in_Lakhs'].mean().sort_values(ascending=False).round(2))

# EDA 16: Owner type
print("\n[EDA 16] Owner Type distribution:")
print(df['Owner_Type'].value_counts())

# EDA 17: Availability status
print("\n[EDA 17] Availability Status distribution:")
print(df['Availability_Status'].value_counts())

# EDA 18: Parking vs Price
print("\n[EDA 18] Avg Price by Parking Space:")
print(df.groupby('Parking_Space')['Price_in_Lakhs'].mean().sort_values(ascending=False).round(2))

# EDA 20: Transport vs investment
print("\n[EDA 20] Good Investment rate by Transport Accessibility:")
print(df.groupby('Public_Transport_Accessibility')['Good_Investment'].mean().round(3))

print("\n✅ EDA complete. Processed data saved.")
