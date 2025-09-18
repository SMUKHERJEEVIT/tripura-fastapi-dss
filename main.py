# main.py
import pandas as pd
import joblib
import json
from fastapi import FastAPI
from pydantic import BaseModel

# --- Pydantic Model for Input Validation ---
# This ensures that any request to your endpoint MUST contain a district and a village.
class RegionFilter(BaseModel):
    district: str
    village: str

# --- Configuration & Asset Loading ---
MODEL_PATH = 'ml_model.joblib'
SCALER_PATH = 'scaler.joblib'
COLUMNS_PATH = 'model_columns.json'
RAW_DATA_PATH = 'tripura_household_data.csv'
MEDIANS_PATH = 'imputation_medians.json'

# --- Initialize FastAPI App ---
app = FastAPI(title="Tripura DSS API")

# --- Load all assets at startup ---
try:
    ml_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    with open(COLUMNS_PATH, 'r') as f:
        model_columns_info = json.load(f)
        model_columns = model_columns_info['all_columns']
        numeric_columns_to_scale = model_columns_info['numeric_columns']
        model_feature_cols = model_columns_info['original_feature_columns']
        
    with open(MEDIANS_PATH, 'r') as f:
        imputation_medians = json.load(f)

    raw_df = pd.read_csv(RAW_DATA_PATH)
    print("All ML assets and data loaded successfully.")
    
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find a required file: {e}")
    ml_model = None

# --- Preprocessing function (remains the same) ---
def preprocess_input(input_df_raw):
    # This function's internal logic is identical to the Flask version
    if not ml_model:
        raise RuntimeError("Model is not loaded. Cannot preprocess.")
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

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
def read_root():
    return {"message": "Welcome to the Tripura DSS API Engine"}

# The endpoint now uses the Pydantic model `RegionFilter` for the request body.
# FastAPI handles the JSON parsing and validation automatically.
@app.post("/api/dss/predict_region")
def predict_region(filters: RegionFilter):
    if not ml_model:
        return {"error": "Server is not ready, model not loaded."}

    # Access data via attributes (filters.district) instead of dictionary keys (filters['district'])
    region_df = raw_df[(raw_df['district'] == filters.district) & (raw_df['village'] == filters.village)].copy()
    if region_df.empty:
        return {"error": "No households found for the selected region"}

    processed_df = preprocess_input(region_df)
    ml_predictions = ml_model.predict(processed_df)
    ml_probabilities = ml_model.predict_proba(processed_df)
    
    labels = ['has_pm_kisan', 'has_jal_jeevan', 'has_mgnrega']
    results = []
    
    for idx, (original_index, row) in enumerate(region_df.iterrows()):
        recommendations = {}
        for j, scheme in enumerate(labels):
            recommendations[scheme] = {
                'eligible': bool(ml_predictions[idx, j]), 
                'confidence_score': float(ml_probabilities[j][idx, 1]), 
                'source': 'ML Model'
            }
        
        if row['water_index'] < 0.3 and row['soil_index'] < 0.4:
            recommendations['has_jal_jeevan'].update({
                'eligible': True, 
                'confidence_score': 1.0, 
                'source': 'Rule-Based Override (Low Water & Soil Index)'
            })
        
        results.append({
            "patta_holder_name": row['patta_holder_name'], 
            "recommendations": recommendations
        })
        
    return {"region": filters.dict(), "results": results}
