"""
Step 2: EDA Visualizations
Generates and saves all 20 EDA charts as PNG files.
Run: python step2_eda_plots.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'C:\Users\SURYAPRAKASH JHA\Downloads\real_estate_project\real_estate_project\data\processed_data.csv')
OUT = 'eda_outputs'
import os; os.makedirs(OUT, exist_ok=True)

PALETTE = ['#2196F3','#FF5722','#4CAF50','#9C27B0','#FF9800']

def savefig(name):
    plt.tight_layout()
    plt.savefig(f'{OUT}/{name}.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUT}/{name}.png")

print("Generating EDA plots...")

# ── EDA 1: Price Distribution ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(df['Price_in_Lakhs'], bins=60, color=PALETTE[0], edgecolor='white', linewidth=0.4)
ax.axvline(df['Price_in_Lakhs'].mean(),   color='red',    linestyle='--', label=f"Mean ₹{df['Price_in_Lakhs'].mean():.1f}L")
ax.axvline(df['Price_in_Lakhs'].median(), color='orange', linestyle='--', label=f"Median ₹{df['Price_in_Lakhs'].median():.1f}L")
ax.set_title('EDA 1 — Distribution of Property Prices', fontsize=14, fontweight='bold')
ax.set_xlabel('Price (₹ Lakhs)'); ax.set_ylabel('Count')
ax.legend(); savefig('eda01_price_distribution')

# ── EDA 2: Size Distribution ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(df['Size_in_SqFt'], bins=60, color=PALETTE[2], edgecolor='white', linewidth=0.4)
ax.axvline(df['Size_in_SqFt'].mean(),   color='red',    linestyle='--', label=f"Mean {df['Size_in_SqFt'].mean():.0f} sqft")
ax.axvline(df['Size_in_SqFt'].median(), color='orange', linestyle='--', label=f"Median {df['Size_in_SqFt'].median():.0f} sqft")
ax.set_title('EDA 2 — Distribution of Property Size', fontsize=14, fontweight='bold')
ax.set_xlabel('Size (SqFt)'); ax.set_ylabel('Count')
ax.legend(); savefig('eda02_size_distribution')

# ── EDA 3: Price/SqFt by Property Type ───────────────────────
fig, ax = plt.subplots(figsize=(8,5))
data = df.groupby('Property_Type')['Price_per_SqFt'].mean().sort_values(ascending=False)
bars = ax.bar(data.index, data.values, color=PALETTE[:len(data)], edgecolor='white')
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{b.get_height():.3f}',
            ha='center', va='bottom', fontsize=10)
ax.set_title('EDA 3 — Avg Price/SqFt by Property Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Price per SqFt'); savefig('eda03_price_per_sqft_by_type')

# ── EDA 4: Size vs Price scatter ─────────────────────────────
sample = df.sample(3000, random_state=42)
fig, ax = plt.subplots(figsize=(10,6))
sc = ax.scatter(sample['Size_in_SqFt'], sample['Price_in_Lakhs'],
                c=sample['BHK'], cmap='viridis', alpha=0.4, s=10)
plt.colorbar(sc, ax=ax, label='BHK')
ax.set_title('EDA 4 — Size vs Price (colored by BHK)', fontsize=14, fontweight='bold')
ax.set_xlabel('Size (SqFt)'); ax.set_ylabel('Price (₹ Lakhs)')
savefig('eda04_size_vs_price')

# ── EDA 5: Outliers boxplot ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12,5))
axes[0].boxplot(df['Price_in_Lakhs'], patch_artist=True,
                boxprops=dict(facecolor=PALETTE[0], alpha=0.6))
axes[0].set_title('Price (₹ Lakhs)'); axes[0].set_ylabel('Value')
axes[1].boxplot(df['Size_in_SqFt'], patch_artist=True,
                boxprops=dict(facecolor=PALETTE[2], alpha=0.6))
axes[1].set_title('Size (SqFt)')
fig.suptitle('EDA 5 — Outlier Detection (Box Plots)', fontsize=14, fontweight='bold')
savefig('eda05_outliers_boxplot')

# ── EDA 6: Avg Price/SqFt by State ───────────────────────────
fig, ax = plt.subplots(figsize=(14,6))
state_p = df.groupby('State')['Price_in_Lakhs'].mean().sort_values(ascending=False)
ax.bar(range(len(state_p)), state_p.values, color=PALETTE[0], edgecolor='white')
ax.set_xticks(range(len(state_p))); ax.set_xticklabels(state_p.index, rotation=45, ha='right', fontsize=8)
ax.set_title('EDA 6 — Avg Property Price by State', fontsize=14, fontweight='bold')
ax.set_ylabel('Avg Price (₹ Lakhs)'); savefig('eda06_price_by_state')

# ── EDA 7: Avg Price by City (top 20) ────────────────────────
fig, ax = plt.subplots(figsize=(14,6))
city_p = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(20)
colors = [PALETTE[0] if v >= city_p.median() else PALETTE[1] for v in city_p.values]
ax.bar(range(len(city_p)), city_p.values, color=colors, edgecolor='white')
ax.set_xticks(range(len(city_p))); ax.set_xticklabels(city_p.index, rotation=45, ha='right', fontsize=8)
ax.set_title('EDA 7 — Top 20 Cities by Avg Property Price', fontsize=14, fontweight='bold')
ax.set_ylabel('Avg Price (₹ Lakhs)'); savefig('eda07_price_by_city')

# ── EDA 8: Median Age by Locality (top 15) ───────────────────
fig, ax = plt.subplots(figsize=(14,6))
loc_age = df.groupby('Locality')['Age_of_Property'].median().sort_values(ascending=False).head(15)
ax.barh(range(len(loc_age)), loc_age.values, color=PALETTE[3], edgecolor='white')
ax.set_yticks(range(len(loc_age))); ax.set_yticklabels(loc_age.index, fontsize=8)
ax.set_title('EDA 8 — Median Property Age by Locality (Top 15)', fontsize=14, fontweight='bold')
ax.set_xlabel('Median Age (Years)'); savefig('eda08_age_by_locality')

# ── EDA 9: BHK distribution across cities ────────────────────
top6 = df['City'].value_counts().head(6).index
bhk_city = df[df['City'].isin(top6)].groupby(['City','BHK']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(12,6))
bhk_city.plot(kind='bar', ax=ax, colormap='tab10', edgecolor='white')
ax.set_title('EDA 9 — BHK Distribution in Top 6 Cities', fontsize=14, fontweight='bold')
ax.set_xlabel('City'); ax.set_ylabel('Count')
ax.legend(title='BHK', bbox_to_anchor=(1.01,1))
ax.tick_params(axis='x', rotation=30); savefig('eda09_bhk_by_city')

# ── EDA 10: Price trends top 5 localities ────────────────────
top5_loc = df.groupby('Locality')['Price_in_Lakhs'].mean().nlargest(5).index
fig, ax = plt.subplots(figsize=(12,6))
for i, loc in enumerate(top5_loc):
    sub = df[df['Locality'] == loc].groupby('Year_Built')['Price_in_Lakhs'].mean()
    ax.plot(sub.index, sub.values, marker='o', markersize=3,
            label=loc, color=PALETTE[i % len(PALETTE)])
ax.set_title('EDA 10 — Price Trends for Top 5 Localities', fontsize=14, fontweight='bold')
ax.set_xlabel('Year Built'); ax.set_ylabel('Avg Price (₹ Lakhs)')
ax.legend(fontsize=8); savefig('eda10_price_trends_top_localities')

# ── EDA 11: Correlation Heatmap ───────────────────────────────
num_cols = ['Price_in_Lakhs','Size_in_SqFt','BHK','Age_of_Property',
            'Nearby_Schools','Nearby_Hospitals','Infrastructure_Score',
            'Future_Price_5Y','Good_Investment']
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10,8))
im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols, fontsize=8)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=7,
                color='black' if abs(corr.iloc[i,j]) < 0.7 else 'white')
ax.set_title('EDA 11 — Feature Correlation Heatmap', fontsize=14, fontweight='bold')
savefig('eda11_correlation_heatmap')

# ── EDA 12: Schools vs Price ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10,5))
school_price = df.groupby('Nearby_Schools')['Price_in_Lakhs'].mean()
ax.bar(school_price.index, school_price.values, color=PALETTE[1], edgecolor='white')
ax.set_title('EDA 12 — Nearby Schools vs Avg Price', fontsize=14, fontweight='bold')
ax.set_xlabel('Nearby Schools'); ax.set_ylabel('Avg Price (₹ Lakhs)'); savefig('eda12_schools_vs_price')

# ── EDA 13: Hospitals vs Price ────────────────────────────────
fig, ax = plt.subplots(figsize=(10,5))
hosp_price = df.groupby('Nearby_Hospitals')['Price_in_Lakhs'].mean()
ax.bar(hosp_price.index, hosp_price.values, color=PALETTE[2], edgecolor='white')
ax.set_title('EDA 13 — Nearby Hospitals vs Avg Price', fontsize=14, fontweight='bold')
ax.set_xlabel('Nearby Hospitals'); ax.set_ylabel('Avg Price (₹ Lakhs)'); savefig('eda13_hospitals_vs_price')

# ── EDA 14: Price by Furnished Status ────────────────────────
fig, ax = plt.subplots(figsize=(8,5))
furn_p = df.groupby('Furnished_Status')['Price_in_Lakhs'].mean().sort_values(ascending=False)
bars = ax.bar(furn_p.index, furn_p.values, color=PALETTE[:len(furn_p)], edgecolor='white')
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'₹{b.get_height():.1f}L',
            ha='center', va='bottom', fontsize=10)
ax.set_title('EDA 14 — Avg Price by Furnished Status', fontsize=14, fontweight='bold')
ax.set_ylabel('Avg Price (₹ Lakhs)'); savefig('eda14_price_by_furnished')

# ── EDA 15: Price/SqFt by Facing Direction ───────────────────
fig, ax = plt.subplots(figsize=(10,5))
facing_p = df.groupby('Facing')['Price_per_SqFt'].mean().sort_values(ascending=False)
ax.bar(facing_p.index, facing_p.values, color=PALETTE[4], edgecolor='white')
ax.set_title('EDA 15 — Avg Price/SqFt by Facing Direction', fontsize=14, fontweight='bold')
ax.set_ylabel('Price per SqFt'); ax.tick_params(axis='x', rotation=30); savefig('eda15_price_by_facing')

# ── EDA 16: Owner Type Distribution ──────────────────────────
fig, ax = plt.subplots(figsize=(7,5))
ot = df['Owner_Type'].value_counts()
wedge_props = dict(width=0.5, edgecolor='white', linewidth=2)
ax.pie(ot.values, labels=ot.index, autopct='%1.1f%%', colors=PALETTE[:len(ot)],
       wedgeprops=wedge_props, startangle=90)
ax.set_title('EDA 16 — Owner Type Distribution', fontsize=14, fontweight='bold')
savefig('eda16_owner_type')

# ── EDA 17: Availability Status ───────────────────────────────
fig, ax = plt.subplots(figsize=(7,5))
av = df['Availability_Status'].value_counts()
ax.pie(av.values, labels=av.index, autopct='%1.1f%%', colors=[PALETTE[0], PALETTE[1]],
       wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2), startangle=90)
ax.set_title('EDA 17 — Availability Status', fontsize=14, fontweight='bold')
savefig('eda17_availability_status')

# ── EDA 18: Parking vs Price ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7,5))
pk = df.groupby('Parking_Space')['Price_in_Lakhs'].mean().sort_values(ascending=False)
bars = ax.bar(pk.index, pk.values, color=PALETTE[:len(pk)], edgecolor='white')
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'₹{b.get_height():.1f}L',
            ha='center', va='bottom', fontsize=10)
ax.set_title('EDA 18 — Parking Space vs Avg Price', fontsize=14, fontweight='bold')
ax.set_ylabel('Avg Price (₹ Lakhs)'); savefig('eda18_parking_vs_price')

# ── EDA 19: Amenities vs Price ────────────────────────────────
fig, ax = plt.subplots(figsize=(9,5))
am = df.groupby('Amenities')['Price_per_SqFt'].mean().sort_values(ascending=False)
ax.barh(range(len(am)), am.values, color=PALETTE[3], edgecolor='white')
ax.set_yticks(range(len(am))); ax.set_yticklabels(am.index, fontsize=8)
ax.set_title('EDA 19 — Amenities vs Avg Price/SqFt', fontsize=14, fontweight='bold')
ax.set_xlabel('Avg Price per SqFt'); savefig('eda19_amenities_vs_price')

# ── EDA 20: Transport Accessibility vs Investment ─────────────
fig, axes = plt.subplots(1, 2, figsize=(12,5))
tr_price = df.groupby('Public_Transport_Accessibility')['Price_in_Lakhs'].mean().sort_values(ascending=False)
axes[0].bar(tr_price.index, tr_price.values, color=PALETTE[:3], edgecolor='white')
axes[0].set_title('Avg Price by Transport Access', fontsize=11)
axes[0].set_ylabel('Avg Price (₹ Lakhs)')

tr_gi = df.groupby('Public_Transport_Accessibility')['Good_Investment'].mean().mul(100).sort_values(ascending=False)
axes[1].bar(tr_gi.index, tr_gi.values, color=PALETTE[2], edgecolor='white')
axes[1].set_title('Good Investment % by Transport Access', fontsize=11)
axes[1].set_ylabel('Good Investment %')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())

fig.suptitle('EDA 20 — Transport Accessibility vs Price & Investment', fontsize=13, fontweight='bold')
savefig('eda20_transport_vs_investment')

print(f"\n✅ All 20 EDA plots saved to '{OUT}/' folder.")
