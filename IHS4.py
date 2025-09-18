import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
from flask import Flask, request, jsonify
import json
import os

# --- Configuration ---
RAW_DATA_PATH = 'tripura_household_data.csv' 
FEATURES_PATH = 'train_features.csv'
LABELS_PATH = 'train_labels.csv'
MODEL_PATH = 'ml_model.joblib'
SCALER_PATH = 'scaler.joblib'
COLUMNS_PATH = 'model_columns.json'
TRIPURA_GEO_FILE = 'tripura_geo_data.csv' 

print("Starting the ML + DSS Engine Development Process for Tripura...")

# ==============================================================================
# Step 1: Data Collection (Enhanced with Realistic Index Ranges)
# ==============================================================================
print("\n### Step 1: Data Collection ###")

def collect_data(num_records=1000):
    """
    Simulates fetching data for Tripura and creates a raw DataFrame.
    MODIFIED: Now uses realistic, location-specific index ranges for data generation.
    """
    try:
        geo_df = pd.read_csv(TRIPURA_GEO_FILE)
        location_choices = list(zip(geo_df['District'], geo_df['Village']))
        print(f"Successfully loaded {len(location_choices)} locations from '{TRIPURA_GEO_FILE}'.")
    except Exception as e:
        print(f"Error: Could not read '{TRIPURA_GEO_FILE}'. Please ensure it exists. Error: {e}")
        location_choices = [('West Tripura', 'Agartala'), ('Gomati', 'Udaipur')]

    # NEW: Dictionary of realistic index ranges per location
    index_ranges = {
        ('West Tripura', 'Agartala'):   {'veg': (0.20, 0.45), 'soil': (0.25, 0.40), 'water': (0.10, 0.30)},
        ('West Tripura', 'Jirania'):    {'veg': (0.35, 0.60), 'soil': (0.30, 0.55), 'water': (0.20, 0.45)},
        ('West Tripura', 'Mohanpur'):   {'veg': (0.40, 0.70), 'soil': (0.40, 0.65), 'water': (0.30, 0.50)},
        ('Gomati', 'Udaipur'):          {'veg': (0.30, 0.65), 'soil': (0.45, 0.70), 'water': (0.40, 0.70)},
        ('Gomati', 'Amarpur'):          {'veg': (0.55, 0.80), 'soil': (0.50, 0.70), 'water': (0.35, 0.60)},
        ('Gomati', 'Karbook'):          {'veg': (0.60, 0.85), 'soil': (0.45, 0.65), 'water': (0.25, 0.50)},
        ('Dhalai', 'Ambassa'):          {'veg': (0.65, 0.90), 'soil': (0.50, 0.75), 'water': (0.30, 0.55)},
        ('Dhalai', 'Kamalpur'):         {'veg': (0.50, 0.80), 'soil': (0.45, 0.70), 'water': (0.30, 0.50)},
        ('Dhalai', 'Manu'):             {'veg': (0.60, 0.85), 'soil': (0.50, 0.70), 'water': (0.25, 0.45)},
        ('Sepahijala', 'Bishalgarh'):   {'veg': (0.40, 0.70), 'soil': (0.40, 0.65), 'water': (0.30, 0.60)},
        ('Sepahijala', 'Sonamura'):     {'veg': (0.35, 0.60), 'soil': (0.55, 0.80), 'water': (0.50, 0.80)},
    }
    default_ranges = {'veg': (0.2, 0.9), 'soil': (0.1, 0.8), 'water': (0.1, 0.9)}

    np.random.seed(42)
    chosen_locations = [location_choices[i] for i in np.random.randint(0, len(location_choices), num_records)]
    
    # MODIFIED: Generate indices based on the specific ranges for each chosen location
    vegetation_indices = []
    soil_indices = []
    water_indices = []
    for district, village in chosen_locations:
        ranges = index_ranges.get((district, village), default_ranges)
        vegetation_indices.append(np.random.uniform(ranges['veg'][0], ranges['veg'][1]))
        soil_indices.append(np.random.uniform(ranges['soil'][0], ranges['soil'][1]))
        water_indices.append(np.random.uniform(ranges['water'][0], ranges['water'][1]))

    data = {
        'household_id': range(1, num_records + 1),
        'patta_holder_name': [f'Holder_{i}' for i in range(1, num_records + 1)],
        'state': ['Tripura'] * num_records,
        'district': [loc[0] for loc in chosen_locations],
        'village': [loc[1] for loc in chosen_locations],
        'tribal_group': np.random.choice(['Tripuri', 'Reang', 'Jamatia'], num_records),
        'claim_type': np.random.choice(['IFR', 'CR', 'CFR'], num_records),
        'claim_status': np.random.choice(['Approved', 'Pending'], num_records, p=[0.9, 0.1]),
        'land_area': np.random.uniform(0.5, 10, num_records),
        'pond_count': np.random.randint(0, 4, num_records),
        'homestead_area': np.random.uniform(0.1, 0.5, num_records),
        'vegetation_index': vegetation_indices,
        'soil_index': soil_indices,
        'water_index': water_indices,
        'forest_cover_pct': np.random.uniform(10, 80, num_records),
        'dist_to_road_km': np.random.uniform(0.2, 5, num_records),
        'has_pm_kisan': np.random.choice([0, 1], num_records, p=[0.6, 0.4]),
        'has_jal_jeevan': np.random.choice([0, 1], num_records, p=[0.7, 0.3]),
        'has_mgnrega': np.random.choice([0, 1], num_records, p=[0.5, 0.5])
    }
    df = pd.DataFrame(data)
    df.loc[df.sample(frac=0.05, random_state=1).index, 'vegetation_index'] = np.nan
    df.loc[df.sample(frac=0.05, random_state=2).index, 'soil_index'] = np.nan
    return df

raw_df = collect_data()
raw_df.to_csv(RAW_DATA_PATH, index=False)
print("Raw Tripura data sample with realistic, location-specific indices:\n", raw_df.head())

# ==============================================================================
# Step 2: Data Preprocessing
# ==============================================================================
print("\n### Step 2: Data Preprocessing ###")
df = pd.read_csv(RAW_DATA_PATH)
imputation_medians = {
    'vegetation_index': df['vegetation_index'].median(), 
    'soil_index': df['soil_index'].median()
}
for col, median_val in imputation_medians.items():
    df[col] = df[col].fillna(median_val)

labels = ['has_pm_kisan', 'has_jal_jeevan', 'has_mgnrega']
model_feature_cols = [
    'land_area', 'pond_count', 'homestead_area', 'vegetation_index', 'soil_index',
    'water_index', 'forest_cover_pct', 'dist_to_road_km', 'tribal_group'
]
features = df[model_feature_cols]
y = df[labels]
features = pd.get_dummies(features, columns=['tribal_group'], drop_first=True) 

# ==============================================================================
# Step 3: Feature Engineering & Scaling
# ==============================================================================
print("\n### Step 3: Feature Engineering & Scaling ###")
features['forest_dependency_index'] = features['forest_cover_pct'] / (features['land_area'] + 1e-6)
numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
scaler = MinMaxScaler()
features[numeric_cols] = scaler.fit_transform(features[numeric_cols])
features.to_csv(FEATURES_PATH, index=False)
y.to_csv(LABELS_PATH, index=False)
joblib.dump(scaler, SCALER_PATH)
model_columns_info = {'all_columns': list(features.columns), 'numeric_columns': numeric_cols}
with open(COLUMNS_PATH, 'w') as f:
    json.dump(model_columns_info, f)
print("Preprocessed data, scaler, and column info saved.")

# ==============================================================================
# Step 4 & 5: Model Training
# ==============================================================================
print("\n### Step 4 & 5: Model Training ###")
X_train, X_val, y_train, y_val = train_test_split(features, y, test_size=0.2, random_state=42)
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
multi_output_model = MultiOutputClassifier(estimator=xgb_classifier)
multi_output_model.fit(X_train, y_train)
y_pred = multi_output_model.predict(X_val)
for i, label in enumerate(labels):
    f1 = f1_score(y_val.iloc[:, i], y_pred[:, i], average='binary')
    print(f"F1-Score for {label}: {f1:.4f}")
joblib.dump(multi_output_model, MODEL_PATH)
print(f"Trained model saved to '{MODEL_PATH}'")

# ==============================================================================
# Step 6 & 8: Model Integration with ENHANCED DSS Backend
# ==============================================================================
print("\n### Step 6 & 8: Model Integration with Enhanced DSS Backend ###")
app = Flask(__name__)
ml_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(COLUMNS_PATH, 'r') as f:
    model_columns_info = json.load(f)
    model_columns = model_columns_info['all_columns']
    numeric_columns_to_scale = model_columns_info['numeric_columns']

def preprocess_input(input_df_raw):
    input_df = input_df_raw.copy()
    input_df = input_df[[col for col in model_feature_cols if col in input_df.columns]]
    for col, median_val in imputation_medians.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].fillna(median_val)
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    input_df['forest_dependency_index'] = input_df['forest_cover_pct'] / (input_df['land_area'] + 1e-6)
    if not input_df.empty and numeric_columns_to_scale:
        input_df[numeric_columns_to_scale] = scaler.transform(input_df[numeric_columns_to_scale])
    input_df = input_df[model_columns]
    return input_df

@app.route("/api/dss/get_holders", methods=['POST'])
def get_holders():
    filters = request.get_json()
    filtered_df = raw_df[(raw_df['district'] == filters.get('district')) & (raw_df['village'] == filters.get('village'))].copy()
    for col, median_val in imputation_medians.items():
        if col in filtered_df.columns:
            filtered_df[col] = filtered_df[col].fillna(median_val)
    result_cols = ['patta_holder_name', 'claim_type', 'land_area', 'claim_status']
    return jsonify(filtered_df[result_cols].to_dict(orient='records'))

@app.route("/api/dss/region_summary", methods=['POST'])
def region_summary():
    filters = request.get_json()
    filtered_df = raw_df[(raw_df['district'] == filters.get('district')) & (raw_df['village'] == filters.get('village'))]
    if filtered_df.empty: return jsonify({"error": "No data for selected region"}), 404
    summary = {
        "vegetation_index_avg": filtered_df['vegetation_index'].mean(),
        "soil_index_avg": filtered_df['soil_index'].mean(),
        "water_index_avg": filtered_df['water_index'].mean(),
        "agricultural_index_avg": filtered_df['land_area'].mean(),
        "forest_cover_avg_pct": filtered_df['forest_cover_pct'].mean(),
        "infrastructure_access_avg_km": filtered_df['dist_to_road_km'].mean(),
        "total_households": len(filtered_df)
    }
    return jsonify(summary)

@app.route("/api/dss/predict_region", methods=['POST'])
def predict_region():
    filters = request.get_json()
    region_df = raw_df[(raw_df['district'] == filters.get('district')) & (raw_df['village'] == filters.get('village'))].copy()
    if region_df.empty: return jsonify({"error": "No households found for the selected region"}), 404
    processed_df = preprocess_input(region_df)
    ml_predictions = ml_model.predict(processed_df)
    ml_probabilities = ml_model.predict_proba(processed_df)
    results = []
    for idx, (i, row) in enumerate(region_df.iterrows()):
        recommendations = {}
        for j, scheme in enumerate(labels):
            recommendations[scheme] = {'eligible': bool(ml_predictions[idx, j]), 'confidence_score': float(ml_probabilities[j][idx, 1]), 'source': 'ML Model'}
        
        if row['water_index'] < 0.3 and row['soil_index'] < 0.4:
            recommendations['has_jal_jeevan'].update({'eligible': True, 'confidence_score': 1.0, 'source': 'Rule-Based Override (Low Water & Soil Index)'})
        
        results.append({"patta_holder_name": row['patta_holder_name'], "recommendations": recommendations})
    return jsonify({"region": filters, "results": results})

print("Flask API app with enhanced endpoints is defined and ready.")

# ==============================================================================
# Step 7: Frontend Integration (Simulation)
# ==============================================================================
print("\n### Step 7: Frontend Integration (Simulation) ###")
test_region = {"district": raw_df['district'].iloc[0], "village": raw_df['village'].iloc[0]}

def simulate_api_call(endpoint, data):
    with app.test_request_context(endpoint, method='POST', data=json.dumps(data), content_type='application/json'):
        response = app.full_dispatch_request()
        return json.loads(response.data)

print("\n--- Simulating UI Action: Applying Filters to view 'FRA Holder Data Viewer' ---")
holder_data = simulate_api_call('/api/dss/get_holders', test_region)
print(json.dumps(holder_data[:3], indent=2))

print("\n--- Simulating UI Action: Loading the 'Asset Intelligence Panel' ---")
asset_summary = simulate_api_call('/api/dss/region_summary', test_region)
print(json.dumps(asset_summary, indent=2))

print(f"\n--- Simulating UI Action: Clicking 'Run DSS Engine' for the Region ---")
dss_results = simulate_api_call('/api/dss/predict_region', test_region)
dss_results['results'] = dss_results['results'][:3] 
print(json.dumps(dss_results, indent=2))

# ==============================================================================
# Step 9: Testing & Validation
# ==============================================================================
print("\n### Step 9: Testing & Validation ###")

def validate_model_performance(model, X_val, y_val):
    """Tests the model against the validation set."""
    predictions = model.predict(X_val)
    print("Validation against unseen data:")
    for i, label in enumerate(labels):
        f1 = f1_score(y_val.iloc[:, i], predictions[:, i])
        print(f"  - F1-Score for {label}: {f1:.4f}")

validate_model_performance(multi_output_model, X_val, y_val)

# ==============================================================================
# Step 10: Documentation for Team
# ==============================================================================
print("\n### Step 10: Documentation for Team ###")
documentation_summary = f"""
--------------------------------------------------------------------------------
ML+DSS ENGINE DOCUMENTATION (TRIPURA)
--------------------------------------------------------------------------------
1. Data Sources: Simulated data for Tripura from '{RAW_DATA_PATH}'
   - Key Features: vegetation_index, soil_index, water_index.
   - NOTE: Data generation uses realistic, location-specific index ranges.
2. API Endpoints:
   - POST /api/dss/get_holders: Gets holder details for a region.
   - POST /api/dss/region_summary: Gets asset summary for a region.
   - POST /api/dss/predict_region: Runs the DSS engine for a region.
--------------------------------------------------------------------------------
"""
print(documentation_summary)
# Import necessary libraries - pip install pandas numpy scikit-learn xgboost Flask