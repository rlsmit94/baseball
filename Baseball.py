import os
import pymssql
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_baseball_data(query):
    conn = pymssql.connect(
        server='localhost',
        port=1434,
        user='sa',
        password=os.getenv('BASEBALL_DB_PASSWORD'),
        database='BaseballDB'
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Get comprehensive batting data for three true outcomes analysis
query = """
SELECT yearID,
       playerID,
       teamID,
       HR,      -- Home runs
       SO,      -- Strikeouts  
       BB,      -- Walks (bases on balls)
       AB,      -- At bats
       H,       -- Hits
       HBP,     -- Hit by pitch
       SF       -- Sacrifice flies
FROM BaseballDB.dbo.Batting 
WHERE AB >= 200  -- Players with significant playing time
  AND HR IS NOT NULL 
  AND SO IS NOT NULL
  AND BB IS NOT NULL
  AND yearID >= 1950  -- Focus on modern era with complete data
ORDER BY yearID, playerID
"""

# Get the data
data = get_baseball_data(query)
print(f"Total qualified player seasons: {len(data)}")
print(f"Data spans: {data['yearID'].min()} to {data['yearID'].max()}")

# Calculate rates and additional metrics
data['PA'] = data['AB'] + data['BB'] + data['HBP'].fillna(0) + data['SF'].fillna(0)  # Plate appearances
data['bb_rate'] = data['BB'] / data['PA'] * 100  # Walk rate per 100 PA
data['so_rate'] = data['SO'] / data['PA'] * 100  # Strikeout rate per 100 PA  
data['hr_rate'] = data['HR'] / data['PA'] * 100  # Home run rate per 100 PA
data['three_outcomes_rate'] = (data['HR'] + data['SO'] + data['BB']) / data['PA'] * 100  # Combined rate
data['contact_rate'] = ((data['AB'] - data['SO']) / data['AB']) * 100  # Contact rate

# Define eras for analysis
def assign_era(year):
    if year < 1961:
        return "Pre-Expansion"
    elif year < 1977:
        return "Expansion"
    elif year < 1995:
        return "Modern"
    elif year < 2006:
        return "Steroid"
    else:
        return "Analytics"

data['era'] = data['yearID'].apply(assign_era)

# Remove outliers and missing values
clean_data = data.dropna(subset=['bb_rate', 'so_rate', 'hr_rate']).copy()
# Remove extreme outliers (99th percentile)
for col in ['bb_rate', 'so_rate', 'hr_rate']:
    q99 = clean_data[col].quantile(0.99)
    clean_data = clean_data[clean_data[col] <= q99]

print(f"Clean dataset for regression: {len(clean_data)} observations")
print(f"Average three true outcomes rate: {clean_data['three_outcomes_rate'].mean():.1f}%")

# Manual multiple regression function
def multiple_regression(X, y, feature_names):
    """Perform multiple linear regression"""
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    try:
        # Calculate coefficients: β = (X'X)^-1 X'y
        XTX = np.dot(X_with_intercept.T, X_with_intercept)
        XTX_inv = np.linalg.inv(XTX)
        XTy = np.dot(X_with_intercept.T, y)
        coefficients = np.dot(XTX_inv, XTy)
        
        # Predictions and residuals
        y_pred = np.dot(X_with_intercept, coefficients)
        residuals = y - y_pred
        
        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Standard errors and t-stats
        mse = ss_res / (len(y) - len(coefficients))
        var_covar_matrix = mse * XTX_inv
        std_errors = np.sqrt(np.diag(var_covar_matrix))
        t_stats = coefficients / std_errors
        
        return {
            'coefficients': coefficients,
            'std_errors': std_errors,
            't_stats': t_stats,
            'r_squared': r_squared,
            'predictions': y_pred,
            'residuals': residuals,
            'feature_names': ['Intercept'] + feature_names
        }
    except np.linalg.LinAlgError:
        print("Matrix inversion failed")
        return None

print("\n" + "="*80)
print("THREE TRUE OUTCOMES REGRESSION ANALYSIS")
print("="*80)

# Model 1: Walks predicted by HR and SO
print("\nMODEL 1: BB_RATE ~ HR_RATE + SO_RATE")
print("-" * 50)

X1 = clean_data[['hr_rate', 'so_rate']].values
y1 = clean_data['bb_rate'].values
model1 = multiple_regression(X1, y1, ['HR_Rate', 'SO_Rate'])

if model1:
    print("Results:")
    for name, coef, se, t_stat in zip(model1['feature_names'], model1['coefficients'], 
                                      model1['std_errors'], model1['t_stats']):
        sig = "***" if abs(t_stat) > 2.58 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.645 else ""
        print(f"  {name:12}: {coef:8.4f} (SE: {se:.4f}, t: {t_stat:6.2f}) {sig}")
    
    print(f"\nR-squared: {model1['r_squared']:.4f}")
    print(f"Interpretation:")
    print(f"  • Each 1% increase in HR rate → {model1['coefficients'][1]:+.3f}% change in BB rate")
    print(f"  • Each 1% increase in SO rate → {model1['coefficients'][2]:+.3f}% change in BB rate")

# Model 2: Add time trend and player context
print("\nMODEL 2: BB_RATE ~ HR_RATE + SO_RATE + YEAR + CONTACT_RATE")
print("-" * 60)

clean_data['year_norm'] = clean_data['yearID'] - clean_data['yearID'].min()
X2 = clean_data[['hr_rate', 'so_rate', 'year_norm', 'contact_rate']].values
model2 = multiple_regression(X2, y1, ['HR_Rate', 'SO_Rate', 'Year', 'Contact_Rate'])

if model2:
    print("Results:")
    for name, coef, se, t_stat in zip(model2['feature_names'], model2['coefficients'], 
                                      model2['std_errors'], model2['t_stats']):
        sig = "***" if abs(t_stat) > 2.58 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.645 else ""
        print(f"  {name:12}: {coef:8.4f} (SE: {se:.4f}, t: {t_stat:6.2f}) {sig}")
    
    print(f"\nR-squared: {model2['r_squared']:.4f}")
    print(f"Improvement over Model 1: {model2['r_squared'] - model1['r_squared']:.4f}")

# Model 3: Era-specific analysis
print("\nMODEL 3: ERA-SPECIFIC RELATIONSHIPS")
print("-" * 40)

era_results = {}
for era in clean_data['era'].unique():
    era_data = clean_data[clean_data['era'] == era]
    if len(era_data) > 100:  # Minimum sample size
        X_era = era_data[['hr_rate', 'so_rate']].values
        y_era = era_data['bb_rate'].values
        model_era = multiple_regression(X_era, y_era, ['HR_Rate', 'SO_Rate'])
        
        if model_era:
            era_results[era] = model_era
            print(f"\n{era} Era ({len(era_data)} players):")
            print(f"  BB_Rate = {model_era['coefficients'][0]:.2f} + {model_era['coefficients'][1]:.4f}*HR + {model_era['coefficients'][2]:.4f}*SO")
            print(f"  R²: {model_era['r_squared']:.3f}")
            print(f"  HR effect: {model_era['coefficients'][1]:.4f} (t: {model_era['t_stats'][1]:.2f})")
            print(f"  SO effect: {model_era['coefficients'][2]:.4f} (t: {model_era['t_stats'][2]:.2f})")

# Create comprehensive visualizations
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Plot 1: BB vs HR scatter with regression line
axes[0, 0].scatter(clean_data['hr_rate'], clean_data['bb_rate'], alpha=0.5, s=15, color='blue')
if model1:
    hr_range = np.linspace(clean_data['hr_rate'].min(), clean_data['hr_rate'].max(), 100)
    # For visualization, hold SO at median
    so_median = clean_data['so_rate'].median()
    bb_pred = model1['coefficients'][0] + model1['coefficients'][1] * hr_range + model1['coefficients'][2] * so_median
    axes[0, 0].plot(hr_range, bb_pred, 'r-', linewidth=2, 
                    label=f'Slope: {model1["coefficients"][1]:.3f}')
    axes[0, 0].legend()
axes[0, 0].set_xlabel('Home Run Rate (%)')
axes[0, 0].set_ylabel('Walk Rate (%)')
axes[0, 0].set_title('Walk Rate vs Home Run Rate')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: BB vs SO scatter
axes[0, 1].scatter(clean_data['so_rate'], clean_data['bb_rate'], alpha=0.5, s=15, color='green')
if model1:
    so_range = np.linspace(clean_data['so_rate'].min(), clean_data['so_rate'].max(), 100)
    hr_median = clean_data['hr_rate'].median()
    bb_pred = model1['coefficients'][0] + model1['coefficients'][1] * hr_median + model1['coefficients'][2] * so_range
    axes[0, 1].plot(so_range, bb_pred, 'r-', linewidth=2, 
                    label=f'Slope: {model1["coefficients"][2]:.3f}')
    axes[0, 1].legend()
axes[0, 1].set_xlabel('Strikeout Rate (%)')
axes[0, 1].set_ylabel('Walk Rate (%)')
axes[0, 1].set_title('Walk Rate vs Strikeout Rate')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Three True Outcomes rate over time
yearly_tto = clean_data.groupby('yearID')['three_outcomes_rate'].mean()
axes[1, 0].plot(yearly_tto.index, yearly_tto.values, 'purple', linewidth=3, marker='o')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Three True Outcomes Rate (%)')
axes[1, 0].set_title('Evolution of Three True Outcomes')
axes[1, 0].grid(True, alpha=0.3)

# Add trend line
years = yearly_tto.index.values
rates = yearly_tto.values
z = np.polyfit(years, rates, 1)
p = np.poly1d(z)
axes[1, 0].plot(years, p(years), "r--", linewidth=2, 
                label=f'Trend: {z[0]:.3f}% per year')
axes[1, 0].legend()

# Plot 4: Residuals vs Fitted (Model 2)
if model2:
    axes[1, 1].scatter(model2['predictions'], model2['residuals'], alpha=0.5, s=15)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Fitted (Model 2)')
    axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Era comparison - coefficients
if era_results:
    era_names = list(era_results.keys())
    hr_effects = [era_results[era]['coefficients'][1] for era in era_names]
    so_effects = [era_results[era]['coefficients'][2] for era in era_names]
    
    x = np.arange(len(era_names))
    width = 0.35
    
    bars1 = axes[2, 0].bar(x - width/2, hr_effects, width, label='HR Effect', alpha=0.7, color='red')
    bars2 = axes[2, 0].bar(x + width/2, so_effects, width, label='SO Effect', alpha=0.7, color='blue')
    
    axes[2, 0].set_xlabel('Era')
    axes[2, 0].set_ylabel('Effect on Walk Rate')
    axes[2, 0].set_title('HR and SO Effects on Walks by Era')
    axes[2, 0].set_xticks(x)
    axes[2, 0].set_xticklabels(era_names, rotation=45)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Plot 6: Distribution by era
era_data_list = []
era_labels = []
for era in ['Pre-Expansion', 'Expansion', 'Modern', 'Steroid', 'Analytics']:
    if era in clean_data['era'].values:
        era_subset = clean_data[clean_data['era'] == era]
        era_data_list.append(era_subset['three_outcomes_rate'].values)
        era_labels.append(f"{era}\n(n={len(era_subset)})")

if era_data_list:
    bp = axes[2, 1].boxplot(era_data_list, tick_labels=era_labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    axes[2, 1].set_ylabel('Three True Outcomes Rate (%)')
    axes[2, 1].set_title('TTO Distribution by Era')
    axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('three_true_outcomes_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary insights
print("\n" + "="*80)
print("KEY INSIGHTS: THREE TRUE OUTCOMES ANALYSIS")
print("="*80)

if model1 and model2:
    print(f"\n1. CORE RELATIONSHIPS:")
    hr_effect = model1['coefficients'][1]
    so_effect = model1['coefficients'][2]
    
    print(f"   • HR-BB relationship: {'Positive' if hr_effect > 0 else 'Negative'} ({hr_effect:+.4f})")
    print(f"   • SO-BB relationship: {'Positive' if so_effect > 0 else 'Negative'} ({so_effect:+.4f})")
    
    if hr_effect > 0 and so_effect > 0:
        print(f"   • Players who hit more HRs AND strike out more ALSO walk more")
        print(f"   • This supports the 'three true outcomes' philosophy")
    elif hr_effect > 0 and so_effect < 0:
        print(f"   • Power hitters walk more, but contact hitters (low SO) walk less")
    
    print(f"\n2. STATISTICAL SIGNIFICANCE:")
    hr_sig = "Yes" if abs(model1['t_stats'][1]) > 1.96 else "No"
    so_sig = "Yes" if abs(model1['t_stats'][2]) > 1.96 else "No"
    print(f"   • HR effect significant: {hr_sig} (t = {model1['t_stats'][1]:.2f})")
    print(f"   • SO effect significant: {so_sig} (t = {model1['t_stats'][2]:.2f})")

print(f"\n4. ERA EVOLUTION:")
if era_results_bb and era_results_so:
    print("   • How the relationships have changed over time:")
    for era in era_results_bb.keys():
        hr_bb = era_results_bb[era]['coefficients'][1]
        so_bb = era_results_bb[era]['coefficients'][2] 
        hr_so = era_results_so[era]['coefficients'][1]
        bb_so = era_results_so[era]['coefficients'][2]
        print(f"     - {era}: HR→BB = {hr_bb:+.4f}, SO→BB = {so_bb:+.4f}, HR→SO = {hr_so:+.4f}, BB→SO = {bb_so:+.4f}")

print(f"\n5. PRACTICAL IMPLICATIONS:")
print(f"   • Three True Outcomes rate has {'increased' if z[0] > 0 else 'decreased'} by {abs(z[0]):.3f}% per year")
total_change = z[0] * (clean_data['yearID'].max() - clean_data['yearID'].min())
print(f"   • Total change over study period: {total_change:+.1f} percentage points")

# Calculate correlation between all three outcomes
hr_bb_corr = clean_data['hr_rate'].corr(clean_data['bb_rate'])
hr_so_corr = clean_data['hr_rate'].corr(clean_data['so_rate'])
bb_so_corr = clean_data['bb_rate'].corr(clean_data['so_rate'])

print(f"\n6. SIMPLE CORRELATIONS (for comparison):")
print(f"   • HR-BB correlation: {hr_bb_corr:.4f}")
print(f"   • HR-SO correlation: {hr_so_corr:.4f}")  
print(f"   • BB-SO correlation: {bb_so_corr:.4f}")

if hr_bb_corr > 0 and hr_so_corr > 0 and bb_so_corr > 0:
    print(f"   • All three outcomes are positively correlated!")
    print(f"   • This is the statistical proof of 'three true outcomes' baseball")
elif any([hr_bb_corr > 0.3, hr_so_corr > 0.3, bb_so_corr > 0.3]):
    print(f"   • Strong positive relationships exist between most outcomes")
    print(f"   • Modern baseball philosophy has statistical validity")

print(f"\n7. THE THREE TRUE OUTCOMES VERDICT:")
avg_tto_early = clean_data[clean_data['yearID'] <= 1970]['three_outcomes_rate'].mean()
avg_tto_recent = clean_data[clean_data['yearID'] >= 2000]['three_outcomes_rate'].mean()
tto_increase = avg_tto_recent - avg_tto_early

print(f"   • Early era (≤1970) TTO rate: {avg_tto_early:.1f}%")
print(f"   • Recent era (≥2000) TTO rate: {avg_tto_recent:.1f}%") 
print(f"   • Increase: {tto_increase:+.1f} percentage points")
print(f"   • The 'three true outcomes' approach has {'clearly emerged' if tto_increase > 5 else 'somewhat increased' if tto_increase > 2 else 'remained stable'} in modern baseball")