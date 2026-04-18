"""
Real Estate Investment Advisor — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# ─── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── LOAD MODELS & METADATA ───────────────────────────────────
@st.cache_resource
def load_models():
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

@st.cache_data
def load_data():
    return pd.read_csv('data/processed_data.csv')

clf, reg, scaler, meta, exp_log = load_models()
df = load_data()

FEATURES = meta['features']

# ─── HELPER ───────────────────────────────────────────────────
def safe_enc(mapping, key, default=0):
    return int(mapping.get(str(key), default))

def make_input_row(city, state, prop_type, bhk, size_sqft, price_lakhs,
                   floor_no, total_floors, year_built, furnished,
                   nearby_schools, nearby_hospitals, transport,
                   parking, security, amenities, facing, owner_type, availability):

    age = 2025 - year_built
    price_per_sqft = (price_lakhs * 1e5) / size_sqft
    school_density = nearby_schools / 10.0
    hospital_density = nearby_hospitals / 10.0

    transport_map = {'High': 3, 'Medium': 2, 'Low': 1}
    parking_map   = {'Yes': 2, 'No': 0}
    security_map  = {'Gated + CCTV + Guard': 3, 'Gated + CCTV': 2, 'Gated': 1, 'None': 0}
    amenity_map   = {'Gym + Pool + Clubhouse': 3, 'Gym + Pool': 2, 'Gym': 1, 'None': 0}

    transport_score = transport_map.get(transport, 2)
    parking_numeric = parking_map.get(parking, 1)
    security_score  = security_map.get(security, 1)
    amenity_score   = amenity_map.get(amenities, 1)

    infra_score = (transport_score + parking_numeric + security_score +
                   amenity_score + school_density + hospital_density)

    city_median = meta['city_medians'].get(city, df['Price_in_Lakhs'].median())
    price_vs_median = price_lakhs / city_median

    is_ready = 1 if availability == 'Ready_to_Move' else 0

    prop_enc  = safe_enc(meta['property_type_enc_map'], prop_type)
    furn_enc  = safe_enc(meta['furnished_enc_map'], furnished)
    city_enc  = safe_enc(meta['city_enc_map'], city)
    state_enc = safe_enc(meta['state_enc_map'], state)

    facing_map = df[['Facing','Facing_Enc']].drop_duplicates().set_index('Facing')['Facing_Enc'].to_dict()
    owner_map  = df[['Owner_Type','Owner_Type_Enc']].drop_duplicates().set_index('Owner_Type')['Owner_Type_Enc'].to_dict()
    avail_map  = df[['Availability_Status','Availability_Status_Enc']].drop_duplicates().set_index('Availability_Status')['Availability_Status_Enc'].to_dict()

    row = {
        'BHK': bhk,
        'Size_in_SqFt': size_sqft,
        'Price_in_Lakhs': price_lakhs,
        'Floor_No': floor_no,
        'Total_Floors': total_floors,
        'Age_of_Property': age,
        'Nearby_Schools': nearby_schools,
        'Nearby_Hospitals': nearby_hospitals,
        'Price_per_SqFt_Calc': price_per_sqft,
        'School_Density_Score': school_density,
        'Hospital_Density_Score': hospital_density,
        'Infrastructure_Score': infra_score,
        'Is_Ready_to_Move': is_ready,
        'Transport_Score': transport_score,
        'Parking_Numeric': parking_numeric,
        'Security_Score': security_score,
        'Amenity_Score': amenity_score,
        'Price_vs_CityMedian': price_vs_median,
        'Property_Type_Enc': prop_enc,
        'Furnished_Status_Enc': furn_enc,
        'Facing_Enc': facing_map.get(facing, 0),
        'Owner_Type_Enc': owner_map.get(owner_type, 0),
        'Availability_Status_Enc': avail_map.get(availability, 0),
        'City_Enc': city_enc,
        'State_Enc': state_enc,
    }
    return pd.DataFrame([row])[FEATURES]

# ─── SIDEBAR ──────────────────────────────────────────────────
st.sidebar.title("🏠 Navigation")
page = st.sidebar.radio("Go to", [
    "🔮 Investment Predictor",
    "📊 Market Explorer",
    "📈 EDA Insights",
    "🧪 Model Performance"
])

# ══════════════════════════════════════════════════════════════
# PAGE 1: PREDICTOR
# ══════════════════════════════════════════════════════════════
if page == "🔮 Investment Predictor":
    st.title("🏠 Real Estate Investment Advisor")
    st.markdown("##### Predict if a property is a **Good Investment** and estimate its **5-Year Future Price**")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Location & Property")
        cities = sorted(meta['cities'])
        city = st.selectbox("City", cities)
        # Auto-select state
        state_for_city = df[df['City'] == city]['State'].mode()[0]
        st.info(f"State: **{state_for_city}**")
        state = state_for_city

        prop_type = st.selectbox("Property Type", ['Apartment', 'Independent House', 'Villa'])
        availability = st.selectbox("Availability", ['Ready_to_Move', 'Under_Construction'])
        owner_type = st.selectbox("Owner Type", ['Owner', 'Builder', 'Broker'])
        facing = st.selectbox("Facing", sorted(df['Facing'].unique().tolist()))

    with col2:
        st.subheader("🏗️ Property Details")
        bhk = st.slider("BHK", 1, 5, 3)
        size_sqft = st.slider("Size (SqFt)", 500, 5000, 1500, step=50)
        price_lakhs = st.number_input("Price (₹ Lakhs)", min_value=10.0, max_value=500.0,
                                       value=150.0, step=5.0)
        year_built = st.slider("Year Built", 1990, 2024, 2010)
        floor_no = st.slider("Floor Number", 0, 30, 3)
        total_floors = st.slider("Total Floors in Building", 1, 40, 10)

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🏫 Neighbourhood")
        nearby_schools    = st.slider("Nearby Schools", 0, 10, 3)
        nearby_hospitals  = st.slider("Nearby Hospitals", 0, 10, 2)
        transport = st.selectbox("Public Transport Access", ['High', 'Medium', 'Low'])

    with col4:
        st.subheader("🏢 Amenities & Security")
        furnished = st.selectbox("Furnished Status", ['Unfurnished', 'Semi-furnished', 'Furnished'])
        parking   = st.selectbox("Parking", ['Yes', 'No'])
        security  = st.selectbox("Security", ['Gated + CCTV + Guard', 'Gated + CCTV', 'Gated', 'None'])
        amenities = st.selectbox("Amenities", ['Gym + Pool + Clubhouse', 'Gym + Pool', 'Gym', 'None'])

    st.divider()
    predict_btn = st.button("🔮 Predict Investment Value", type="primary", use_container_width=True)

    if predict_btn:
        input_df = make_input_row(
            city, state, prop_type, bhk, size_sqft, price_lakhs,
            floor_no, total_floors, year_built, furnished,
            nearby_schools, nearby_hospitals, transport,
            parking, security, amenities, facing, owner_type, availability
        )

        # Classification
        clf_pred  = clf.predict(input_df)[0]
        clf_proba = clf.predict_proba(input_df)[0]
        confidence = clf_proba[clf_pred] * 100

        # Regression
        future_price = reg.predict(input_df)[0]
        appreciation = ((future_price - price_lakhs) / price_lakhs) * 100
        annual_return = (((future_price / price_lakhs) ** (1/5)) - 1) * 100

        # Display results
        r1, r2, r3 = st.columns(3)

        with r1:
            if clf_pred == 1:
                st.success("✅ **GOOD INVESTMENT**")
                st.metric("Confidence", f"{confidence:.1f}%")
            else:
                st.error("❌ **NOT A GOOD INVESTMENT**")
                st.metric("Confidence", f"{confidence:.1f}%")

        with r2:
            st.info(f"### ₹ {future_price:.1f} Lakhs\n**Estimated Price After 5 Years**")

        with r3:
            st.metric("Total Appreciation", f"+{appreciation:.1f}%",
                      delta=f"~{annual_return:.1f}% / year")

        st.divider()

        # Price context
        city_median = meta['city_medians'].get(city, df['Price_in_Lakhs'].median())
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"₹ {price_lakhs:.1f} L")
        c2.metric(f"City Median ({city})", f"₹ {city_median:.1f} L",
                  delta=f"{'Below' if price_lakhs <= city_median else 'Above'} median")
        c3.metric("Price per SqFt", f"₹ {(price_lakhs*1e5/size_sqft):,.0f}")

        # Recommendation text
        st.subheader("📋 Investment Summary")
        growth_rate = meta['city_growth_rates'].get(city, 0.085)
        lines = [
            f"- **Location**: {city}, {state}",
            f"- **Property**: {bhk}BHK {prop_type}, {size_sqft} SqFt",
            f"- **Current Price**: ₹{price_lakhs:.1f} Lakhs (₹{price_lakhs*100000/size_sqft:,.0f}/sqft)",
            f"- **City Growth Rate Applied**: {growth_rate*100:.1f}% p.a.",
            f"- **5-Year Forecast**: ₹{future_price:.1f} Lakhs ({appreciation:+.1f}% total)",
            f"- **Annual Return**: ~{annual_return:.1f}%",
            f"- **Verdict**: {'✅ Recommended for investment' if clf_pred == 1 else '⚠️ Consider alternatives or negotiate price'}",
        ]
        st.markdown("\n".join(lines))

# ══════════════════════════════════════════════════════════════
# PAGE 2: MARKET EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📊 Market Explorer":
    st.title("📊 Market Explorer")
    st.markdown("Filter and explore properties across cities.")
    st.divider()

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        sel_cities = st.multiselect("City", sorted(df['City'].unique()), default=["Mumbai", "Bangalore", "Chennai"]
                                    if "Mumbai" in df['City'].values else sorted(df['City'].unique())[:3])
    with f2:
        price_range = st.slider("Price Range (₹ Lakhs)", 10, 500, (50, 300))
    with f3:
        bhk_filter = st.multiselect("BHK", [1,2,3,4,5], default=[2,3])

    prop_filter = st.multiselect("Property Type", df['Property_Type'].unique().tolist(),
                                  default=df['Property_Type'].unique().tolist())

    filtered = df.copy()
    if sel_cities:
        filtered = filtered[filtered['City'].isin(sel_cities)]
    filtered = filtered[
        (filtered['Price_in_Lakhs'] >= price_range[0]) &
        (filtered['Price_in_Lakhs'] <= price_range[1])
    ]
    if bhk_filter:
        filtered = filtered[filtered['BHK'].isin(bhk_filter)]
    if prop_filter:
        filtered = filtered[filtered['Property_Type'].isin(prop_filter)]

    st.metric("Matching Properties", f"{len(filtered):,}")

    if len(filtered) > 0:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Price", f"₹ {filtered['Price_in_Lakhs'].mean():.1f} L")
        m2.metric("Median Price", f"₹ {filtered['Price_in_Lakhs'].median():.1f} L")
        m3.metric("Good Investment %", f"{filtered['Good_Investment'].mean()*100:.1f}%")
        m4.metric("Avg Size", f"{filtered['Size_in_SqFt'].mean():.0f} SqFt")

        st.divider()

        # City price comparison
        st.subheader("City-wise Average Price")
        city_avg = filtered.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).reset_index()
        city_avg.columns = ['City', 'Avg Price (Lakhs)']
        st.bar_chart(city_avg.set_index('City'))

        # BHK distribution
        st.subheader("BHK Distribution")
        bhk_dist = filtered['BHK'].value_counts().sort_index().reset_index()
        bhk_dist.columns = ['BHK', 'Count']
        st.bar_chart(bhk_dist.set_index('BHK'))

        # Good investment by city
        st.subheader("Good Investment Rate by City")
        gi_city = filtered.groupby('City')['Good_Investment'].mean().mul(100).sort_values(ascending=False).reset_index()
        gi_city.columns = ['City', 'Good Investment %']
        st.bar_chart(gi_city.set_index('City'))

        # Sample data
        st.subheader("Sample Properties")
        display_cols = ['City', 'Property_Type', 'BHK', 'Size_in_SqFt',
                        'Price_in_Lakhs', 'Furnished_Status', 'Good_Investment', 'Future_Price_5Y']
        sample = filtered[display_cols].sample(min(20, len(filtered)), random_state=42)
        sample['Good_Investment'] = sample['Good_Investment'].map({1:'✅ Yes', 0:'❌ No'})
        sample['Future_Price_5Y'] = sample['Future_Price_5Y'].round(1)
        st.dataframe(sample, use_container_width=True)
    else:
        st.warning("No properties match the current filters.")

# ══════════════════════════════════════════════════════════════
# PAGE 3: EDA INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "📈 EDA Insights":
    st.title("📈 Exploratory Data Analysis")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Price Analysis", "Location Analysis",
                                       "Feature Relationships", "Investment Analysis"])

    with tab1:
        st.subheader("Price Distribution by Property Type")
        pt_price = df.groupby('Property_Type')['Price_in_Lakhs'].agg(['mean','median','min','max']).round(2)
        st.dataframe(pt_price, use_container_width=True)

        st.subheader("Price Distribution Stats")
        price_bins = pd.cut(df['Price_in_Lakhs'], bins=[0,100,200,300,400,500],
                            labels=['0-100L','100-200L','200-300L','300-400L','400-500L'])
        price_dist = price_bins.value_counts().sort_index().reset_index()
        price_dist.columns = ['Range', 'Count']
        st.bar_chart(price_dist.set_index('Range'))

        st.subheader("Size Distribution Stats")
        size_bins = pd.cut(df['Size_in_SqFt'], bins=[0,1000,2000,3000,4000,5000],
                           labels=['<1K','1K-2K','2K-3K','3K-4K','4K-5K'])
        size_dist = size_bins.value_counts().sort_index().reset_index()
        size_dist.columns = ['Size Range', 'Count']
        st.bar_chart(size_dist.set_index('Size Range'))

    with tab2:
        st.subheader("Top 15 Cities by Avg Price (₹ Lakhs)")
        city_p = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(15).reset_index()
        city_p.columns = ['City', 'Avg Price']
        st.bar_chart(city_p.set_index('City'))

        st.subheader("Top States by Avg Price (₹ Lakhs)")
        state_p = df.groupby('State')['Price_in_Lakhs'].mean().sort_values(ascending=False).reset_index()
        state_p.columns = ['State', 'Avg Price']
        st.bar_chart(state_p.set_index('State'))

        st.subheader("BHK Distribution Across Top Cities")
        top5_cities = df['City'].value_counts().head(5).index.tolist()
        bhk_city = df[df['City'].isin(top5_cities)].groupby(['City','BHK']).size().unstack(fill_value=0)
        st.dataframe(bhk_city, use_container_width=True)

    with tab3:
        st.subheader("Avg Price by Furnished Status")
        furn_p = df.groupby('Furnished_Status')['Price_in_Lakhs'].mean().sort_values(ascending=False).reset_index()
        furn_p.columns = ['Furnished Status', 'Avg Price']
        st.bar_chart(furn_p.set_index('Furnished Status'))

        st.subheader("Avg Price by Facing Direction")
        facing_p = df.groupby('Facing')['Price_in_Lakhs'].mean().sort_values(ascending=False).reset_index()
        facing_p.columns = ['Facing', 'Avg Price']
        st.bar_chart(facing_p.set_index('Facing'))

        st.subheader("Infrastructure Score vs Price (Sampled)")
        sample = df.sample(2000, random_state=42)[['Infrastructure_Score','Price_in_Lakhs']]
        sample.columns = ['Infrastructure Score', 'Price (Lakhs)']
        st.scatter_chart(sample, x='Infrastructure Score', y='Price (Lakhs)')

        st.subheader("Feature Correlations with Price")
        num_cols = ['Price_in_Lakhs', 'Size_in_SqFt', 'BHK', 'Age_of_Property',
                    'Nearby_Schools', 'Nearby_Hospitals', 'Infrastructure_Score']
        corr = df[num_cols].corr()[['Price_in_Lakhs']].drop('Price_in_Lakhs').sort_values('Price_in_Lakhs', ascending=False)
        corr.columns = ['Correlation with Price']
        st.dataframe(corr.round(3), use_container_width=True)

    with tab4:
        st.subheader("Good Investment Rate by City")
        gi_city2 = df.groupby('City')['Good_Investment'].mean().mul(100).sort_values(ascending=False).reset_index()
        gi_city2.columns = ['City', '% Good Investment']
        st.bar_chart(gi_city2.set_index('City'))

        st.subheader("Good Investment Rate by Transport Accessibility")
        gi_tr = df.groupby('Public_Transport_Accessibility')['Good_Investment'].mean().mul(100).reset_index()
        gi_tr.columns = ['Transport Accessibility', '% Good Investment']
        st.bar_chart(gi_tr.set_index('Transport Accessibility'))

        st.subheader("Good Investment Rate by BHK")
        gi_bhk = df.groupby('BHK')['Good_Investment'].mean().mul(100).reset_index()
        gi_bhk.columns = ['BHK', '% Good Investment']
        st.bar_chart(gi_bhk.set_index('BHK'))

        st.subheader("Owner Type Distribution")
        ot = df['Owner_Type'].value_counts().reset_index()
        ot.columns = ['Owner Type', 'Count']
        st.dataframe(ot, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "🧪 Model Performance":
    st.title("🧪 Model Performance & Experiment Tracking")
    st.divider()

    st.subheader("🏆 Classification Results — Good Investment Prediction")
    clf_df = pd.DataFrame(exp_log['classification'])
    clf_df['confusion_matrix'] = clf_df['confusion_matrix'].apply(str)
    st.dataframe(clf_df[['model','accuracy','f1_score']], use_container_width=True)

    # Winner highlight
    best_clf_row = clf_df.loc[clf_df['f1_score'].idxmax()]
    st.success(f"✅ Best Classifier: **{best_clf_row['model']}** | Accuracy: {best_clf_row['accuracy']:.4f} | F1: {best_clf_row['f1_score']:.4f}")

    st.subheader("📊 Regression Results — Future Price Prediction")
    reg_df = pd.DataFrame(exp_log['regression'])
    st.dataframe(reg_df[['model','RMSE','MAE','R2']], use_container_width=True)

    best_reg_row = reg_df.loc[reg_df['RMSE'].idxmin()]
    st.success(f"✅ Best Regressor: **{best_reg_row['model']}** | RMSE: {best_reg_row['RMSE']:.4f} L | R²: {best_reg_row['R2']:.4f}")

    st.divider()
    st.subheader("🔢 Feature Importance — Classification (Random Forest)")
    try:
        fi_clf = pd.read_json('models/clf_feature_importance.json', typ='series').sort_values(ascending=False)
        fi_df = fi_clf.reset_index()
        fi_df.columns = ['Feature', 'Importance']
        st.bar_chart(fi_df.set_index('Feature'))
    except Exception:
        st.info("Feature importance file not found.")

    st.subheader("🔢 Feature Importance — Regression (Random Forest)")
    try:
        fi_reg = pd.read_json('models/reg_feature_importance.json', typ='series').sort_values(ascending=False)
        fi_r_df = fi_reg.reset_index()
        fi_r_df.columns = ['Feature', 'Importance']
        st.bar_chart(fi_r_df.set_index('Feature'))
    except Exception:
        st.info("Feature importance file not found.")

    st.divider()
    st.subheader("📦 Model Registry")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Classification Model**
        - Algorithm: Random Forest Classifier
        - n_estimators: 100
        - max_depth: 12
        - Accuracy: 100%
        - F1-Score: 1.0000
        """)
    with col_b:
        st.markdown("""
        **Regression Model**
        - Algorithm: Random Forest Regressor
        - n_estimators: 100
        - max_depth: 12
        - RMSE: 4.20 Lakhs
        - R²: 0.9996
        """)

    st.subheader("📋 Full Experiment Log (JSON)")
    st.json(exp_log)
