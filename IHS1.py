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
STATE_BOUNDARY_FILE = 'state_boundary.csv' # Path to your uploaded file

print("Starting the ML + DSS Engine Development Process...")

# ==============================================================================
# Step 1: Data Collection (Integrated with state_boundary.csv)
# ==============================================================================
print("\n### Step 1: Data Collection ###")

# Goal: Gather data, now using state names from the provided CSV for more realism.

def collect_data(num_records=1000):
    """
    Simulates fetching data and creates a raw DataFrame.
    Integrates state names from the state_boundary.csv file.
    """
    # Load the state boundary data
    try:
        states_df = pd.read_csv(STATE_BOUNDARY_FILE)
        state_names = states_df['state_name'].unique()
        print(f"Successfully loaded {len(state_names)} unique state names from '{STATE_BOUNDARY_FILE}'.")
    except FileNotFoundError:
        print(f"Warning: '{STATE_BOUNDARY_FILE}' not found. Using default placeholder names.")
        state_names = ['State_A', 'State_B', 'State_C']

    np.random.seed(42) # for reproducibility
    data = {
        'household_id': range(1, num_records + 1),
        'state': np.random.choice(state_names, num_records), # Using real state names
        'tribal_group': np.random.choice(['Gond', 'Bhils', 'Santhal'], num_records),
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
    # Introduce some missing values
    df = pd.DataFrame(data)
    df.loc[df.sample(frac=0.05).index, 'land_area'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'water_index'] = np.nan
    return df

raw_df = collect_data()
raw_df.to_csv(RAW_DATA_PATH, index=False)
print(f"Successfully generated and saved raw data to '{RAW_DATA_PATH}'")
print("Raw data sample with integrated state names:\n", raw_df.head())


# ==============================================================================
# Step 2: Data Preprocessing
# ==============================================================================
print("\n### Step 2: Data Preprocessing ###")

df = pd.read_csv(RAW_DATA_PATH)

# Handle missing values
for col in ['land_area', 'water_index']:
    median_val = df[col].median()
    # FIX: Replaced `fillna(inplace=True)` on a slice to avoid FutureWarning
    df[col] = df[col].fillna(median_val)

# Separate features (X) and labels (y)
labels = ['has_pm_kisan', 'has_jal_jeevan', 'has_mgnrega']
features = df.drop(columns=labels + ['household_id'])
y = df[labels]

# Encode categorical features
features = pd.get_dummies(features, columns=['state', 'tribal_group'], drop_first=True)

# ==============================================================================
# Step 3: Feature Engineering & Scaling
# ==============================================================================
print("\n### Step 3: Feature Engineering & Scaling ###")

# Goal: Create more predictive features from existing data.
features['forest_dependency_index'] = features['forest_cover_pct'] / (features['land_area'] + 1e-6)
print("Added 'forest_dependency_index' as a new feature.")

# FIX: Scaling is now done AFTER all features are created.
# Identify numeric columns *after* one-hot encoding and feature engineering
numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
scaler = MinMaxScaler()
features[numeric_cols] = scaler.fit_transform(features[numeric_cols])
print("Numeric features scaled.")

# Save the preprocessed data (deliverables)
features.to_csv(FEATURES_PATH, index=False)
y.to_csv(LABELS_PATH, index=False)

# Save scaler and columns for later use in the API
joblib.dump(scaler, SCALER_PATH)

# FIX: Save both the full column list (for order) and the numeric list (for scaling)
model_columns_info = {
    'all_columns': list(features.columns),
    'numeric_columns': numeric_cols
}
with open(COLUMNS_PATH, 'w') as f:
    json.dump(model_columns_info, f)

print(f"Preprocessed data, scaler, and column info saved.")
print("Final feature matrix sample:\n", features.head())

# ==============================================================================
# Step 4: Model Selection
# ==============================================================================
print("\n### Step 4: Model Selection ###")
print("Model Selected: XGBoost wrapped in MultiOutputClassifier for multi-label classification.")


# ==============================================================================
# Step 5: Model Training
# ==============================================================================
print("\n### Step 5: Model Training ###")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    features, y, test_size=0.2, random_state=42
)

# Initialize and train the model
# FIX: Removed deprecated `use_label_encoder=False` parameter
xgb_classifier = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss',
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
multi_output_model = MultiOutputClassifier(estimator=xgb_classifier)
multi_output_model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the model
y_pred = multi_output_model.predict(X_val)
for i, label in enumerate(labels):
    f1 = f1_score(y_val.iloc[:, i], y_pred[:, i], average='binary')
    print(f"F1-Score for {label}: {f1:.4f}")

# Save the trained model
joblib.dump(multi_output_model, MODEL_PATH)
print(f"Trained model saved to '{MODEL_PATH}'")


# ==============================================================================
# Step 6 & 8: Model Integration with DSS Backend (with Rule-Based Override)
# ==============================================================================

print("\n### Step 6 & 8: Model Integration with DSS Backend ###")

app = Flask(__name__)

# Load artifacts
ml_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(COLUMNS_PATH, 'r') as f:
    # FIX: Load the dictionary containing both all_columns and numeric_columns
    model_columns_info = json.load(f)
    model_columns = model_columns_info['all_columns']
    numeric_columns_to_scale = model_columns_info['numeric_columns']

# Define the global median dictionary for imputation
imputation_medians = {
    'land_area': raw_df['land_area'].median(),
    'water_index': raw_df['water_index'].median()
}


def preprocess_input(input_df):
    """A reusable function to preprocess raw input data for prediction."""
    # Handle missing values using pre-calculated training medians
    for col, median_val in imputation_medians.items():
        if col in input_df.columns:
            # FIX: Avoid FutureWarning by reassigning the column
            input_df[col] = input_df[col].fillna(median_val)

    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df)
    
    # Align columns with the model's training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # Feature Engineering (must match Step 3)
    input_df['forest_dependency_index'] = input_df['forest_cover_pct'] / (input_df['land_area'] + 1e-6)

    # FIX: Normalize ONLY the numeric features the scaler was trained on.
    # This prevents the ValueError.
    input_df[numeric_columns_to_scale] = scaler.transform(input_df[numeric_columns_to_scale])

    # Ensure column order is exactly the same as during training
    input_df = input_df[model_columns]

    return input_df


@app.route("/api/dss/predict", methods=['POST'])
def predict():
    """API endpoint to predict scheme eligibility."""
    input_data = request.get_json()
    household_id = input_data.get('household_id')

    try:
        # NOTE: drop labels before sending to preprocessing
        record = raw_df[raw_df['household_id'] == household_id].drop(columns=labels, errors='ignore')
        if record.empty:
            return jsonify({"error": "Household ID not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    raw_record_for_rules = record.copy()
    processed_record = preprocess_input(record)
    
    # Get ML predictions
    ml_predictions = ml_model.predict(processed_record)[0]
    ml_probabilities = ml_model.predict_proba(processed_record)
    
    recommendations = {}
    for i, scheme in enumerate(labels):
        recommendations[scheme] = {
            'eligible': bool(ml_predictions[i]),
            # Probabilities from predict_proba are structured differently
            'confidence': float(ml_probabilities[i][0, 1]),
            'source': 'ML Model'
        }
        
    # Rule-Based Override
    water_index_val = raw_record_for_rules['water_index'].iloc[0]
    if water_index_val < 0.2:
        recommendations['has_jal_jeevan'].update({
            'eligible': True,
            'confidence': 1.0,
            'source': 'Rule-Based Override (Low Water Index)'
        })
    
    return jsonify({'household_id': household_id, 'recommendations': recommendations})

print("Flask API app is defined and ready.")


# ==============================================================================
# Step 7: Frontend Integration (Simulation)
# ==============================================================================

print("\n### Step 7: Frontend Integration (Simulation) ###")

test_household_id = 42
low_water_household_id = raw_df.sort_values('water_index').iloc[0]['household_id']

def simulate_api_call(household_id):
    """Simulates a client calling the Flask API."""
    with app.test_request_context(
        '/api/dss/predict', method='POST',
        data=json.dumps({'household_id': int(household_id)}), content_type='application/json'
    ):
        response = app.full_dispatch_request()
        return json.loads(response.data)

print(f"\n--- Simulating API call for household_id: {test_household_id} (Normal Case) ---")
print(json.dumps(simulate_api_call(test_household_id), indent=2))

print(f"\n--- Simulating API call for household_id: {low_water_household_id} (Rule Override Case) ---")
print(json.dumps(simulate_api_call(low_water_household_id), indent=2))


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
ML+DSS ENGINE DOCUMENTATION
--------------------------------------------------------------------------------
1. Data Sources: Simulated in `collect_data()`. State names are now sourced
   from '{STATE_BOUNDARY_FILE}' to create a more realistic dataset.

2. Feature Engineering Logic:
   - One-hot encoding for 'state' and 'tribal_group'.
   - A composite feature `forest_dependency_index` was created.
   - MinMax scaling applied to all numeric features *after* all features
     were created to ensure consistency.

3. API Endpoints:
   - Endpoint: POST /api/dss/predict
   - Input JSON:  `{{"household_id": <integer>}}`
   - Output JSON: Contains household_id and recommendations with eligibility,
                  confidence, and source for each scheme.
--------------------------------------------------------------------------------
"""
print(documentation_summary)