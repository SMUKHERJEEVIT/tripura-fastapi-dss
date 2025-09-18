# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import json
import os

# --- Configuration ---
RAW_DATA_PATH = 'tripura_household_data.csv' 
FEATURES_PATH = 'train_features.csv'
LABELS_PATH = 'train_labels.csv'
MODEL_PATH = 'ml_model.joblib'
SCALER_PATH = 'scaler.joblib'
COLUMNS_PATH = 'model_columns.json'
TRIPURA_GEO_FILE = r'C:\Users\soumy\Desktop\Hackatgib\tripura_geo_data.csv' 

def setup_training():
    """Main function to run the entire training pipeline."""
    
    print("Starting the ML Model Training Process for Tripura...")

    # ==============================================================================
    # Step 1: Data Collection
    # ==============================================================================
    print("\n### Step 1: Data Collection ###")

    # NOTE: You must create this CSV file yourself for the script to run.
    # Create a file named 'tripura_geo_data.csv' with two columns: District,Village
    # Example content for tripura_geo_data.csv:
    # District,Village
    # West Tripura,Agartala
    # West Tripura,Jirania
    # Gomati,Udaipur
    # ... and so on for all locations in index_ranges
    
    def collect_data(num_records=1000):
        try:
            geo_df = pd.read_csv(TRIPURA_GEO_FILE)
            location_choices = list(zip(geo_df['District'], geo_df['Village']))
            print(f"Successfully loaded {len(location_choices)} locations from '{TRIPURA_GEO_FILE}'.")
        except Exception as e:
            print(f"Error: Could not read '{TRIPURA_GEO_FILE}'. Please create it. Error: {e}")
            print("Using default locations as a fallback.")
            location_choices = [('West Tripura', 'Agartala'), ('Gomati', 'Udaipur')]

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
        
        vegetation_indices, soil_indices, water_indices = [], [], []
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
    print("Raw Tripura data generated and saved.")

    # ==============================================================================
    # Step 2 & 3: Preprocessing, Feature Engineering & Scaling
    # ==============================================================================
    print("\n### Step 2 & 3: Preprocessing & Feature Engineering ###")
    df = pd.read_csv(RAW_DATA_PATH)
    
    imputation_medians = {
        'vegetation_index': df['vegetation_index'].median(), 
        'soil_index': df['soil_index'].median()
    }
    # Save medians for later use in the API
    with open('imputation_medians.json', 'w') as f:
        json.dump(imputation_medians, f)
        
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
    features['forest_dependency_index'] = features['forest_cover_pct'] / (features['land_area'] + 1e-6)
    
    numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
    scaler = MinMaxScaler()
    features[numeric_cols] = scaler.fit_transform(features[numeric_cols])
    
    features.to_csv(FEATURES_PATH, index=False)
    y.to_csv(LABELS_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)
    
    model_columns_info = {
        'all_columns': list(features.columns), 
        'numeric_columns': numeric_cols,
        'original_feature_columns': model_feature_cols
    }
    with open(COLUMNS_PATH, 'w') as f:
        json.dump(model_columns_info, f)
    print("Preprocessed data, scaler, and column info saved.")

    # ==============================================================================
    # Step 4 & 5: Model Training
    # ==============================================================================
    print("\n### Step 4 & 5: Model Training ###")
    X_train, X_val, y_train, y_val = train_test_split(features, y, test_size=0.2, random_state=42)
    
    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='logloss', 
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42,
    )
    multi_output_model = MultiOutputClassifier(estimator=xgb_classifier)
    multi_output_model.fit(X_train, y_train)
    
    y_pred = multi_output_model.predict(X_val)
    print("Model Performance on Validation Set:")
    for i, label in enumerate(labels):
        f1 = f1_score(y_val.iloc[:, i], y_pred[:, i], average='binary')
        print(f"  - F1-Score for {label}: {f1:.4f}")
        
    joblib.dump(multi_output_model, MODEL_PATH)
    print(f"\nTrained model saved to '{MODEL_PATH}'")
    print("\n--- Training Pipeline Complete! ---")


if __name__ == '__main__':
    setup_training()