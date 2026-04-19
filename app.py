"""
Real Estate Investment Advisor — Streamlit App
Trains models on first run if pkl files are missing (for cloud deployment)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

st.set_page_config(
    page_title="🏠 Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_or_train_models():
    models_ready = (
        os.path.exists('models/best_classifier.pkl') and
        os.path.exists('models/best_regressor.pkl') and
        os.path.exists('models/model_metadata.json')
    )
    if not models_ready:
        st.info("⚙️ First-time setup: training models... (takes ~2 min)")
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        _train_and_save()
    with open('models/best_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('models/best_regressor.pkl', 'rb') as f:
        reg = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/model_metadata.json') as f:
        meta = json.load(f)
    with open('models/experiment_log.json') as f:
        exp_log = json.load(f)
    return clf, reg, scaler, meta, exp_log

def _train_and_save():
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

    import gdown
    if not os.path.exists('india_housing_prices.csv'):
        gdown.download(
            "https://drive.google.com/uc?export=download&id=1eAUbX5-N2WjRMl_rfrgMA3RjN0mCcqDc",
            'india_housing_prices.csv', quiet=False
        )
    df = pd.read_excel('india_housing_prices.xlsx')
    df.drop_duplicates(inplace=True)
    for col in ['Price_in_Lakhs', 'Size_in_SqFt', 'Price_per_SqFt']:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    df['Price_per_SqFt_Calc']    = (df['Price_in_Lakhs'] * 1e5) / df['Size_in_SqFt']
    df['School_Density_Score']   = (df['Nearby_Schools'] - df['Nearby_Schools'].min()) / (df['Nearby_Schools'].max() - df['Nearby_Schools'].min())
    df['Hospital_Density_Score'] = (df['Nearby_Hospitals'] - df['Nearby_Hospitals'].min()) / (df['Nearby_Hospitals'].max() - df['Nearby_Hospitals'].min())

    city_median = df.groupby('City')['Price_in_Lakhs'].transform('median')
    df['Price_vs_CityMedian'] = df['Price_in_Lakhs'] / city_median
    df['Is_Ready_to_Move']    = (df['Availability_Status'] == 'Ready_to_Move').astype(int)

    transport_map = {'High': 3, 'Medium': 2, 'Low': 1}
    parking_map   = {'Yes': 2, 'No': 0, '2': 2, '1': 1, 'None': 0}
    security_map  = {'Gated + CCTV + Guard': 3, 'Gated + CCTV': 2, 'Gated': 1, 'None': 0}
    amenity_map   = {'Gym + Pool + Clubhouse': 3, 'Gym + Pool': 2, 'Gym': 1, 'None': 0}

    df['Transport_Score']      = df['Public_Transport_Accessibility'].map(transport_map)
    df['Parking_Numeric']      = df['Parking_Space'].map(parking_map).fillna(0)
    df['Security_Score']       = df['Security'].map(security_map).fillna(1)
    df['Amenity_Score']        = df['Amenities'].map(amenity_map).fillna(1)
    df['Infrastructure_Score'] = (df['Transport_Score'] + df['Parking_Numeric'] +
                                   df['Security_Score'] + df['Amenity_Score'] +
                                   df['School_Density_Score'] + df['Hospital_Density_Score'])

    city_avg   = df.groupby('City')['Price_in_Lakhs'].mean()
    city_tier  = pd.qcut(city_avg, q=3, labels=['Tier3','Tier2','Tier1'])
    growth_map = {'Tier1': 0.10, 'Tier2': 0.085, 'Tier3': 0.07}
    df['City_Tier']       = df['City'].map(city_tier).astype(str)
    df['Growth_Rate']     = df['City_Tier'].map(growth_map).fillna(0.085)
    df['Future_Price_5Y'] = df['Price_in_Lakhs'] * (1 + df['Growth_Rate']) ** 5

    price_ok = df['Price_vs_CityMedian'] <= 1.0
    sqft_ok  = df['Size_in_SqFt'] >= df['Size_in_SqFt'].median()
    infra_ok = df['Infrastructure_Score'] >= df['Infrastructure_Score'].quantile(0.4)
    df['Good_Investment'] = ((price_ok & infra_ok & (df['BHK'] >= 2)) | (price_ok & sqft_ok)).astype(int)

    le = LabelEncoder()
    label_cols = ['Property_Type','Furnished_Status','Facing','Owner_Type',
                  'Security','Amenities','Public_Transport_Accessibility',
                  'Parking_Space','Availability_Status','City_Tier']
    for col in label_cols:
        df[col + '_Enc'] = le.fit_transform(df[col].astype(str))
    df['City_Enc']  = le.fit_transform(df['City'].astype(str))
    df['State_Enc'] = le.fit_transform(df['State'].astype(str))

    df.to_csv('data/processed_data.csv', index=False)

    FEATURES = [
        'BHK','Size_in_SqFt','Price_in_Lakhs','Floor_No','Total_Floors',
        'Age_of_Property','Nearby_Schools','Nearby_Hospitals',
        'Price_per_SqFt_Calc','School_Density_Score','Hospital_Density_Score',
        'Infrastructure_Score','Is_Ready_to_Move','Transport_Score',
        'Parking_Numeric','Security_Score','Amenity_Score','Price_vs_CityMedian',
        'Property_Type_Enc','Furnished_Status_Enc','Facing_Enc',
        'Owner_Type_Enc','Availability_Status_Enc','City_Enc','State_Enc'
    ]

    X     = df[FEATURES]
    y_cls = df['Good_Investment']
    y_reg = df['Future_Price_5Y']

    X_tr, X_te, yc_tr, yc_te = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    _,    _,    yr_tr, yr_te = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_tr)

    clf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=50, random_state=42, n_jobs=-1)
    clf.fit(X_tr, yc_tr)
    clf_acc = accuracy_score(yc_te, clf.predict(X_te))
    clf_f1  = f1_score(yc_te, clf.predict(X_te))

    reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    reg.fit(X_tr, yr_tr)
    yr_pred  = reg.predict(X_te)
    reg_rmse = float(np.sqrt(mean_squared_error(yr_te, yr_pred)))
    reg_r2   = float(r2_score(yr_te, yr_pred))

    pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False).head(10).to_json('models/clf_feature_importance.json')
    pd.Series(reg.feature_importances_, index=FEATURES).sort_values(ascending=False).head(10).to_json('models/reg_feature_importance.json')

    exp_log = {
        "classification": [{"model": "Random Forest", "accuracy": round(clf_acc,4), "f1_score": round(clf_f1,4)}],
        "regression":     [{"model": "Random Forest", "RMSE": round(reg_rmse,4), "R2": round(reg_r2,4)}]
    }
    with open('models/experiment_log.json', 'w') as f:
        json.dump(exp_log, f, indent=2)

    meta = {
        "features": FEATURES,
        "best_classifier": "Random Forest",
        "best_regressor":  "Random Forest",
        "clf_f1": clf_f1, "reg_rmse": reg_rmse,
        "cities": sorted(df['City'].unique().tolist()),
        "states": sorted(df['State'].unique().tolist()),
        "city_enc_map":          df[['City','City_Enc']].drop_duplicates().set_index('City')['City_Enc'].to_dict(),
        "state_enc_map":         df[['State','State_Enc']].drop_duplicates().set_index('State')['State_Enc'].to_dict(),
        "property_type_enc_map": df[['Property_Type','Property_Type_Enc']].drop_duplicates().set_index('Property_Type')['Property_Type_Enc'].to_dict(),
        "furnished_enc_map":     df[['Furnished_Status','Furnished_Status_Enc']].drop_duplicates().set_index('Furnished_Status')['Furnished_Status_Enc'].to_dict(),
        "city_medians":      df.groupby('City')['Price_in_Lakhs'].median().to_dict(),
        "city_growth_rates": df[['City','Growth_Rate']].drop_duplicates().set_index('City')['Growth_Rate'].to_dict(),
    }
    with open('models/model_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    with open('models/best_classifier.pkl', 'wb') as f: pickle.dump(clf, f)
    with open('models/best_regressor.pkl',  'wb') as f: pickle.dump(reg, f)
    with open('models/scaler.pkl',          'wb') as f: pickle.dump(scaler, f)

@st.cache_data
def load_data():
    if os.path.exists('data/processed_data.csv'):
        return pd.read_csv('data/processed_data.csv')
    return None

clf, reg, scaler, meta, exp_log = load_or_train_models()
df = load_data()
FEATURES = meta['features']

def safe_enc(mapping, key, default=0):
    return int(mapping.get(str(key), default))

def make_input_row(city, state, prop_type, bhk, size_sqft, price_lakhs,
                   floor_no, total_floors, year_built, furnished,
                   nearby_schools, nearby_hospitals, transport,
                   parking, security, amenities, facing, owner_type, availability):
    age              = 2025 - year_built
    price_per_sqft   = (price_lakhs * 1e5) / size_sqft
    school_density   = nearby_schools / 10.0
    hospital_density = nearby_hospitals / 10.0
    transport_score  = {'High': 3, 'Medium': 2, 'Low': 1}.get(transport, 2)
    parking_numeric  = {'Yes': 2, 'No': 0}.get(parking, 1)
    security_score   = {'Gated + CCTV + Guard': 3, 'Gated + CCTV': 2, 'Gated': 1, 'None': 0}.get(security, 1)
    amenity_score    = {'Gym + Pool + Clubhouse': 3, 'Gym + Pool': 2, 'Gym': 1, 'None': 0}.get(amenities, 1)
    infra_score      = transport_score + parking_numeric + security_score + amenity_score + school_density + hospital_density
    city_median      = meta['city_medians'].get(city, 250.0)
    price_vs_median  = price_lakhs / city_median
    is_ready         = 1 if availability == 'Ready_to_Move' else 0
    facing_map = df[['Facing','Facing_Enc']].drop_duplicates().set_index('Facing')['Facing_Enc'].to_dict()
    owner_map  = df[['Owner_Type','Owner_Type_Enc']].drop_duplicates().set_index('Owner_Type')['Owner_Type_Enc'].to_dict()
    avail_map  = df[['Availability_Status','Availability_Status_Enc']].drop_duplicates().set_index('Availability_Status')['Availability_Status_Enc'].to_dict()
    row = {
        'BHK': bhk, 'Size_in_SqFt': size_sqft, 'Price_in_Lakhs': price_lakhs,
        'Floor_No': floor_no, 'Total_Floors': total_floors, 'Age_of_Property': age,
        'Nearby_Schools': nearby_schools, 'Nearby_Hospitals': nearby_hospitals,
        'Price_per_SqFt_Calc': price_per_sqft, 'School_Density_Score': school_density,
        'Hospital_Density_Score': hospital_density, 'Infrastructure_Score': infra_score,
        'Is_Ready_to_Move': is_ready, 'Transport_Score': transport_score,
        'Parking_Numeric': parking_numeric, 'Security_Score': security_score,
        'Amenity_Score': amenity_score, 'Price_vs_CityMedian': price_vs_median,
        'Property_Type_Enc': safe_enc(meta['property_type_enc_map'], prop_type),
        'Furnished_Status_Enc': safe_enc(meta['furnished_enc_map'], furnished),
        'Facing_Enc': facing_map.get(facing, 0),
        'Owner_Type_Enc': owner_map.get(owner_type, 0),
        'Availability_Status_Enc': avail_map.get(availability, 0),
        'City_Enc': safe_enc(meta['city_enc_map'], city),
        'State_Enc': safe_enc(meta['state_enc_map'], state),
    }
    return pd.DataFrame([row])[FEATURES]

st.sidebar.title("🏠 Navigation")
page = st.sidebar.radio("Go to", [
    "🔮 Investment Predictor",
    "📊 Market Explorer",
    "📈 EDA Insights",
    "🧪 Model Performance"
])

if page == "🔮 Investment Predictor":
    st.title("🏠 Real Estate Investment Advisor")
    st.markdown("##### Predict if a property is a **Good Investment** and estimate its **5-Year Future Price**")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 Location & Property")
        city = st.selectbox("City", sorted(meta['cities']))
        state = df[df['City'] == city]['State'].mode()[0]
        st.info(f"State: **{state}**")
        prop_type    = st.selectbox("Property Type", ['Apartment', 'Independent House', 'Villa'])
        availability = st.selectbox("Availability", ['Ready_to_Move', 'Under_Construction'])
        owner_type   = st.selectbox("Owner Type", ['Owner', 'Builder', 'Broker'])
        facing       = st.selectbox("Facing", sorted(df['Facing'].unique().tolist()))
    with col2:
        st.subheader("🏗️ Property Details")
        bhk          = st.slider("BHK", 1, 5, 3)
        size_sqft    = st.slider("Size (SqFt)", 500, 5000, 1500, step=50)
        price_lakhs  = st.number_input("Price (₹ Lakhs)", min_value=10.0, max_value=500.0, value=150.0, step=5.0)
        year_built   = st.slider("Year Built", 1990, 2024, 2010)
        floor_no     = st.slider("Floor Number", 0, 30, 3)
        total_floors = st.slider("Total Floors in Building", 1, 40, 10)
    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🏫 Neighbourhood")
        nearby_schools   = st.slider("Nearby Schools", 0, 10, 3)
        nearby_hospitals = st.slider("Nearby Hospitals", 0, 10, 2)
        transport        = st.selectbox("Public Transport Access", ['High', 'Medium', 'Low'])
    with col4:
        st.subheader("🏢 Amenities & Security")
        furnished = st.selectbox("Furnished Status", ['Unfurnished', 'Semi-furnished', 'Furnished'])
        parking   = st.selectbox("Parking", ['Yes', 'No'])
        security  = st.selectbox("Security", ['Gated + CCTV + Guard', 'Gated + CCTV', 'Gated', 'None'])
        amenities = st.selectbox("Amenities", ['Gym + Pool + Clubhouse', 'Gym + Pool', 'Gym', 'None'])
    st.divider()
    if st.button("🔮 Predict Investment Value", type="primary", use_container_width=True):
        input_df     = make_input_row(city, state, prop_type, bhk, size_sqft, price_lakhs,
                                      floor_no, total_floors, year_built, furnished,
                                      nearby_schools, nearby_hospitals, transport,
                                      parking, security, amenities, facing, owner_type, availability)
        clf_pred     = clf.predict(input_df)[0]
        confidence   = clf.predict_proba(input_df)[0][clf_pred] * 100
        future_price = reg.predict(input_df)[0]
        appreciation = ((future_price - price_lakhs) / price_lakhs) * 100
        annual_return= (((future_price / price_lakhs) ** (1/5)) - 1) * 100
        r1, r2, r3 = st.columns(3)
        with r1:
            st.success("✅ **GOOD INVESTMENT**") if clf_pred == 1 else st.error("❌ **NOT A GOOD INVESTMENT**")
            st.metric("Confidence", f"{confidence:.1f}%")
        with r2:
            st.info(f"### ₹ {future_price:.1f} Lakhs\n**Estimated Price After 5 Years**")
        with r3:
            st.metric("Total Appreciation", f"+{appreciation:.1f}%", delta=f"~{annual_return:.1f}% / year")
        st.divider()
        city_median = meta['city_medians'].get(city, 250.0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"₹ {price_lakhs:.1f} L")
        c2.metric(f"City Median ({city})", f"₹ {city_median:.1f} L",
                  delta=f"{'Below' if price_lakhs <= city_median else 'Above'} median")
        c3.metric("Price per SqFt", f"₹ {(price_lakhs*1e5/size_sqft):,.0f}")
        st.subheader("📋 Investment Summary")
        growth_rate = meta['city_growth_rates'].get(city, 0.085)
        st.markdown("\n".join([
            f"- **Location**: {city}, {state}",
            f"- **Property**: {bhk}BHK {prop_type}, {size_sqft} SqFt",
            f"- **Current Price**: ₹{price_lakhs:.1f} Lakhs",
            f"- **City Growth Rate**: {growth_rate*100:.1f}% p.a.",
            f"- **5-Year Forecast**: ₹{future_price:.1f} Lakhs ({appreciation:+.1f}%)",
            f"- **Annual Return**: ~{annual_return:.1f}%",
            f"- **Verdict**: {'✅ Recommended' if clf_pred == 1 else '⚠️ Consider alternatives or negotiate price'}",
        ]))

elif page == "📊 Market Explorer":
    st.title("📊 Market Explorer")
    st.divider()
    f1, f2, f3 = st.columns(3)
    with f1:
        default_cities = [c for c in ["Mumbai","Bangalore","Chennai"] if c in df['City'].values]
        sel_cities = st.multiselect("City", sorted(df['City'].unique()), default=default_cities or sorted(df['City'].unique())[:3])
    with f2:
        price_range = st.slider("Price Range (₹ Lakhs)", 10, 500, (50, 300))
    with f3:
        bhk_filter = st.multiselect("BHK", [1,2,3,4,5], default=[2,3])
    prop_filter = st.multiselect("Property Type", df['Property_Type'].unique().tolist(), default=df['Property_Type'].unique().tolist())
    filtered = df.copy()
    if sel_cities:  filtered = filtered[filtered['City'].isin(sel_cities)]
    filtered = filtered[(filtered['Price_in_Lakhs'] >= price_range[0]) & (filtered['Price_in_Lakhs'] <= price_range[1])]
    if bhk_filter:  filtered = filtered[filtered['BHK'].isin(bhk_filter)]
    if prop_filter: filtered = filtered[filtered['Property_Type'].isin(prop_filter)]
    st.metric("Matching Properties", f"{len(filtered):,}")
    if len(filtered) > 0:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Price",         f"₹ {filtered['Price_in_Lakhs'].mean():.1f} L")
        m2.metric("Median Price",       f"₹ {filtered['Price_in_Lakhs'].median():.1f} L")
        m3.metric("Good Investment %",  f"{filtered['Good_Investment'].mean()*100:.1f}%")
        m4.metric("Avg Size",           f"{filtered['Size_in_SqFt'].mean():.0f} SqFt")
        st.divider()
        st.subheader("City-wise Average Price")
        st.bar_chart(filtered.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False))
        st.subheader("Good Investment Rate by City")
        st.bar_chart(filtered.groupby('City')['Good_Investment'].mean().mul(100).sort_values(ascending=False))
        st.subheader("Sample Properties")
        display_cols = ['City','Property_Type','BHK','Size_in_SqFt','Price_in_Lakhs','Furnished_Status','Good_Investment','Future_Price_5Y']
        sample = filtered[display_cols].sample(min(20, len(filtered)), random_state=42).copy()
        sample['Good_Investment'] = sample['Good_Investment'].map({1:'✅ Yes', 0:'❌ No'})
        sample['Future_Price_5Y'] = sample['Future_Price_5Y'].round(1)
        st.dataframe(sample, use_container_width=True)
    else:
        st.warning("No properties match the current filters.")

elif page == "📈 EDA Insights":
    st.title("📈 Exploratory Data Analysis")
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["Price Analysis","Location Analysis","Feature Relationships","Investment Analysis"])
    with tab1:
        st.subheader("Price Distribution by Property Type")
        st.dataframe(df.groupby('Property_Type')['Price_in_Lakhs'].agg(['mean','median','min','max']).round(2), use_container_width=True)
        st.subheader("Price Range Distribution")
        price_bins = pd.cut(df['Price_in_Lakhs'], bins=[0,100,200,300,400,500], labels=['0-100L','100-200L','200-300L','300-400L','400-500L'])
        st.bar_chart(price_bins.value_counts().sort_index().rename("Count"))
        st.subheader("Size Range Distribution")
        size_bins = pd.cut(df['Size_in_SqFt'], bins=[0,1000,2000,3000,4000,5000], labels=['<1K','1K-2K','2K-3K','3K-4K','4K-5K'])
        st.bar_chart(size_bins.value_counts().sort_index().rename("Count"))
    with tab2:
        st.subheader("Top 15 Cities by Avg Price")
        st.bar_chart(df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(15))
        st.subheader("States by Avg Price")
        st.bar_chart(df.groupby('State')['Price_in_Lakhs'].mean().sort_values(ascending=False))
        st.subheader("BHK Distribution in Top 6 Cities")
        top6 = df['City'].value_counts().head(6).index
        st.dataframe(df[df['City'].isin(top6)].groupby(['City','BHK']).size().unstack(fill_value=0), use_container_width=True)
    with tab3:
        st.subheader("Price by Furnished Status")
        st.bar_chart(df.groupby('Furnished_Status')['Price_in_Lakhs'].mean().sort_values(ascending=False))
        st.subheader("Price by Facing Direction")
        st.bar_chart(df.groupby('Facing')['Price_in_Lakhs'].mean().sort_values(ascending=False))
        st.subheader("Feature Correlations with Price")
        num_cols = ['Price_in_Lakhs','Size_in_SqFt','BHK','Age_of_Property','Nearby_Schools','Nearby_Hospitals','Infrastructure_Score']
        corr = df[num_cols].corr()[['Price_in_Lakhs']].drop('Price_in_Lakhs').sort_values('Price_in_Lakhs', ascending=False)
        corr.columns = ['Correlation with Price']
        st.dataframe(corr.round(3), use_container_width=True)
    with tab4:
        st.subheader("Good Investment Rate by City")
        st.bar_chart(df.groupby('City')['Good_Investment'].mean().mul(100).sort_values(ascending=False))
        st.subheader("Good Investment Rate by Transport Access")
        st.bar_chart(df.groupby('Public_Transport_Accessibility')['Good_Investment'].mean().mul(100))
        st.subheader("Good Investment Rate by BHK")
        st.bar_chart(df.groupby('BHK')['Good_Investment'].mean().mul(100))

elif page == "🧪 Model Performance":
    st.title("🧪 Model Performance")
    st.divider()
    st.subheader("Classification — Good Investment")
    st.dataframe(pd.DataFrame(exp_log['classification']), use_container_width=True)
    st.subheader("Regression — Future Price (5 Years)")
    st.dataframe(pd.DataFrame(exp_log['regression']), use_container_width=True)
    st.divider()
    st.subheader("Feature Importance — Classification")
    try:
        st.bar_chart(pd.read_json('models/clf_feature_importance.json', typ='series').sort_values(ascending=False))
    except: st.info("Not available.")
    st.subheader("Feature Importance — Regression")
    try:
        st.bar_chart(pd.read_json('models/reg_feature_importance.json', typ='series').sort_values(ascending=False))
    except: st.info("Not available.")
    st.subheader("Full Experiment Log")
    st.json(exp_log)
