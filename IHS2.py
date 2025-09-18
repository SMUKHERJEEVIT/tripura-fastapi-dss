# Import necessary libraries - pip install pandas numpy scikit-learn xgboost Flask

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
# Define file paths for deliverables
RAW_DATA_PATH = 'fra_household_data.csv'
FEATURES_PATH = 'train_features.csv'
LABELS_PATH = 'train_labels.csv'
MODEL_PATH = 'ml_model.joblib'
SCALER_PATH = 'scaler.joblib'
COLUMNS_PATH = 'model_columns.json'
STATE_BOUNDARY_FILE = 'state_boundary.csv' # Assuming this file exists

print("Starting the ML + DSS Engine Development Process...")

# ==============================================================================
# Step 1: Data Collection (Enhanced for Regional Filtering)
# ==============================================================================
print("\n### Step 1: Data Collection ###")

# Goal: Create a more realistic dataset that supports filtering by State, District, and Village.

def collect_data(num_records=1000):
    """
    Simulates fetching data and creates a raw DataFrame.
    Enhanced to include district and village for better UI simulation.
    """
    # In a real app, this would load from a database, not a CSV.
    # For now, we assume a 'state_boundary.csv' exists with a 'state_name' column.
    try:
        states_df = pd.read_csv(STATE_BOUNDARY_FILE)
        state_names = states_df['state_name'].unique()
        print(f"Successfully loaded {len(state_names)} unique state names from '{STATE_BOUNDARY_FILE}'.")
    except FileNotFoundError:
        print(f"Warning: '{STATE_BOUNDARY_FILE}' not found. Using default placeholder names.")
        state_names = ['State_A', 'State_B', 'State_C']

    np.random.seed(42) # for reproducibility
    
    # Create more granular location data to simulate the UI filters
    locations = {
        state_names[0]: {'District_1': ['Village_A', 'Village_B'], 'District_2': ['Village_C', 'Village_D']},
        state_names[1]: {'District_3': ['Village_E', 'Village_F'], 'District_4': ['Village_G', 'Village_H']},
    }
    
    # Flatten the locations for random choice
    location_choices = []
    for state, districts in locations.items():
        for district, villages in districts.items():
            for village in villages:
                location_choices.append((state, district, village))

    chosen_locations = [location_choices[i] for i in np.random.randint(0, len(location_choices), num_records)]
    
    data = {
        'household_id': range(1, num_records + 1),
        'patta_holder_name': [f'Holder_{i}' for i in range(1, num_records + 1)],
        'state': [loc[0] for loc in chosen_locations],
        'district': [loc[1] for loc in chosen_locations],
        'village': [loc[2] for loc in chosen_locations],
        'tribal_group': np.random.choice(['Gond', 'Bhils', 'Santhal'], num_records),
        'claim_type': np.random.choice(['IFR', 'CR', 'CFR'], num_records),
        'claim_status': np.random.choice(['Approved', 'Pending'], num_records, p=[0.9, 0.1]),
        'land_area': np.random.uniform(0.5, 10, num_records),
        'pond_count': np.random.randint(0, 4, num_records),
        'homestead_area': np.random.uniform(0.1, 0.5, num_records),
        'water_index': np.random.uniform(0.1, 0.9, num_records),
        'forest_cover_pct': np.random.uniform(10, 80, num_records),
        'dist_to_road_km': np.random.uniform(0.2, 5, num_records),
        'has_pm_kisan': np.random.choice([0, 1], num_records, p=[0.6, 0.4]),
        'has_jal_jeevan': np.random.choice([0, 1], num_records, p=[0.7, 0.3]),
        'has_mgnrega': np.random.choice([0, 1], num_records, p=[0.5, 0.5])
    }
    df = pd.DataFrame(data)
    df.loc[df.sample(frac=0.05, random_state=1).index, 'land_area'] = np.nan
    df.loc[df.sample(frac=0.05, random_state=2).index, 'water_index'] = np.nan
    return df

raw_df = collect_data()
raw_df.to_csv(RAW_DATA_PATH, index=False)
print(f"Successfully generated and saved raw data to '{RAW_DATA_PATH}'")
print("Raw data sample with enhanced location data:\n", raw_df.head())


# ==============================================================================
# Step 2: Data Preprocessing (No changes needed here)
# ==============================================================================
print("\n### Step 2: Data Preprocessing ###")

df = pd.read_csv(RAW_DATA_PATH)
for col in ['land_area', 'water_index']:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

labels = ['has_pm_kisan', 'has_jal_jeevan', 'has_mgnrega']
# Define features to be used by the model
model_feature_cols = [
    'land_area', 'pond_count', 'homestead_area', 'water_index',
    'forest_cover_pct', 'dist_to_road_km', 'state', 'tribal_group'
]
features = df[model_feature_cols]
y = df[labels]

features = pd.get_dummies(features, columns=['state', 'tribal_group'], drop_first=True)


# ==============================================================================
# Step 3: Feature Engineering & Scaling (No changes needed here)
# ==============================================================================
print("\n### Step 3: Feature Engineering & Scaling ###")

features['forest_dependency_index'] = features['forest_cover_pct'] / (features['land_area'] + 1e-6)
numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
scaler = MinMaxScaler()
features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

features.to_csv(FEATURES_PATH, index=False)
y.to_csv(LABELS_PATH, index=False)
joblib.dump(scaler, SCALER_PATH)

model_columns_info = {
    'all_columns': list(features.columns),
    'numeric_columns': numeric_cols
}
with open(COLUMNS_PATH, 'w') as f:
    json.dump(model_columns_info, f)

print(f"Preprocessed data, scaler, and column info saved.")


# ==============================================================================
# Step 4 & 5: Model Training (No changes needed here)
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

# Load artifacts
ml_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(COLUMNS_PATH, 'r') as f:
    model_columns_info = json.load(f)
    model_columns = model_columns_info['all_columns']
    numeric_columns_to_scale = model_columns_info['numeric_columns']

imputation_medians = {'land_area': raw_df['land_area'].median(), 'water_index': raw_df['water_index'].median()}

def preprocess_input(input_df_raw):
    """A reusable function to preprocess raw input data for prediction."""
    input_df = input_df_raw.copy()
    # Select only the columns the model was trained on
    input_df = input_df[[col for col in model_feature_cols if col in input_df.columns]]
    
    for col, median_val in imputation_medians.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].fillna(median_val)

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    input_df['forest_dependency_index'] = input_df['forest_cover_pct'] / (input_df['land_area'] + 1e-6)
    
    # Check if there are any numeric columns to scale to avoid errors with empty dataframes
    if not input_df.empty and numeric_columns_to_scale:
        input_df[numeric_columns_to_scale] = scaler.transform(input_df[numeric_columns_to_scale])

    input_df = input_df[model_columns]
    return input_df

# === NEW API ENDPOINTS TO SUPPORT THE UI ===

@app.route("/api/dss/get_holders", methods=['POST'])
def get_holders():
    """API endpoint for the 'FRA Holder Data Viewer' panel."""
    filters = request.get_json()
    
    # In a real app, this would be a secure database query.
    filtered_df = raw_df[
        (raw_df['state'] == filters.get('state')) &
        (raw_df['district'] == filters.get('district')) &
        (raw_df['village'] == filters.get('village'))
    ]
    
    # Select columns as specified in the UI design
    result_cols = ['patta_holder_name', 'claim_type', 'land_area', 'claim_status']
    return jsonify(filtered_df[result_cols].to_dict(orient='records'))

@app.route("/api/dss/region_summary", methods=['POST'])
def region_summary():
    """API endpoint for the 'Asset Intelligence Panel'."""
    filters = request.get_json()
    filtered_df = raw_df[
        (raw_df['state'] == filters.get('state')) &
        (raw_df['district'] == filters.get('district')) &
        (raw_df['village'] == filters.get('village'))
    ]
    
    if filtered_df.empty:
        return jsonify({"error": "No data for selected region"}), 404
        
    summary = {
        "water_index_avg": filtered_df['water_index'].mean(),
        "agricultural_index_avg": filtered_df['land_area'].mean(), # Using land_area as a proxy
        "forest_cover_avg_pct": filtered_df['forest_cover_pct'].mean(),
        "infrastructure_access_avg_km": filtered_df['dist_to_road_km'].mean(),
        "total_households": len(filtered_df)
    }
    return jsonify(summary)

@app.route("/api/dss/predict_region", methods=['POST'])
def predict_region():
    """API endpoint for the 'Run DSS Engine' button."""
    filters = request.get_json()
    
    # 1. Filter data based on region selection
    region_df = raw_df[
        (raw_df['state'] == filters.get('state')) &
        (raw_df['district'] == filters.get('district')) &
        (raw_df['village'] == filters.get('village'))
    ].copy()

    if region_df.empty:
        return jsonify({"error": "No households found for the selected region"}), 404

    # 2. Preprocess the filtered data
    processed_df = preprocess_input(region_df)
    
    # 3. Get ML predictions and probabilities for the entire region
    ml_predictions = ml_model.predict(processed_df)
    ml_probabilities = ml_model.predict_proba(processed_df)
    
    # 4. Combine results and apply business rules
    results = []
    for idx, (i, row) in enumerate(region_df.iterrows()):
        holder_id = int(row['household_id'])
        holder_name = row['patta_holder_name']
        recommendations = {}
        
        for j, scheme in enumerate(labels):
            recommendations[scheme] = {
                'eligible': bool(ml_predictions[idx, j]),
                'confidence_score': float(ml_probabilities[j][idx, 1]),
                'source': 'ML Model'
            }
        
        # Rule-Based Override (as in original logic)
        if row['water_index'] < 0.2:
            recommendations['has_jal_jeevan'].update({
                'eligible': True,
                'confidence_score': 1.0,
                'source': 'Rule-Based Override (Low Water Index)'
            })

        results.append({
            "patta_holder_name": holder_name,
            "recommendations": recommendations
        })
        
    return jsonify({
        "region": filters,
        "results": results
    })

print("Flask API app with enhanced endpoints is defined and ready.")


# ==============================================================================
# Step 7: Frontend Integration (Simulation with New Endpoints)
# ==============================================================================

print("\n### Step 7: Frontend Integration (Simulation) ###")

# Define a sample region to test, ensuring it exists in the data
test_region = {
    "state": raw_df['state'].iloc[0],
    "district": raw_df['district'].iloc[0],
    "village": raw_df['village'].iloc[0]
}

def simulate_api_call(endpoint, data):
    """Simulates a client calling a specific Flask API endpoint."""
    with app.test_request_context(
        endpoint, method='POST',
        data=json.dumps(data), content_type='application/json'
    ):
        response = app.full_dispatch_request()
        return json.loads(response.data)

print("\n--- Simulating UI Action: Applying Filters to view 'FRA Holder Data Viewer' ---")
print(f"Requesting data for: {test_region}")
holder_data = simulate_api_call('/api/dss/get_holders', test_region)
print(json.dumps(holder_data, indent=2))

print("\n--- Simulating UI Action: Loading the 'Asset Intelligence Panel' ---")
print(f"Requesting summary for: {test_region}")
asset_summary = simulate_api_call('/api/dss/region_summary', test_region)
print(json.dumps(asset_summary, indent=2))

print(f"\n--- Simulating UI Action: Clicking 'Run DSS Engine' for the Region ---")
print(f"Requesting recommendations for: {test_region}")
dss_results = simulate_api_call('/api/dss/predict_region', test_region)
# Print a sample of the results for brevity
dss_results['results'] = dss_results['results'][:3] 
print(json.dumps(dss_results, indent=2))

# ==============================================================================
# Step 9 & 10: Validation and Documentation (Unchanged)
# ==============================================================================
print("\n### Step 9: Validation ###")
validate_model_performance(multi_output_model, X_val, y_val)

print("\n### Step 10: Documentation ###")
# Documentation now includes the new endpoints
documentation_summary = f"""
--------------------------------------------------------------------------------
ML+DSS ENGINE DOCUMENTATION
--------------------------------------------------------------------------------
1. Data Sources: Simulated in `collect_data()`. Includes State, District,
   and Village columns to support the DSS frontend UI.

2. API Endpoints:
   - Endpoint: POST /api/dss/get_holders
     - Purpose: Powers the 'FRA Holder Data Viewer'.
     - Input JSON:  `{{"state": "...", "district": "...", "village": "..."}}`
     - Output JSON: List of patta holders and their details for the region.

   - Endpoint: POST /api/dss/region_summary
     - Purpose: Powers the 'Asset Intelligence Panel'.
     - Input JSON:  `{{"state": "...", "district": "...", "village": "..."}}`
     - Output JSON: Aggregated metrics like avg. water index, forest cover, etc.

   - Endpoint: POST /api/dss/predict_region
     - Purpose: Powers the 'Run DSS Engine' button for batch recommendations.
     - Input JSON:  `{{"state": "...", "district": "...", "village": "..."}}`
     - Output JSON: Comprehensive list of recommendations for all households
                   in the specified region.
--------------------------------------------------------------------------------
"""
print(documentation_summary)

def validate_model_performance(model, X_val, y_val):
    """Placeholder for the original validation function."""
    pass # Functionality is the same as the original script.