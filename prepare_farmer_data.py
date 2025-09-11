#!/usr/bin/env python3
"""
Prepare Farmer Advisor Dataset for GINN with 3 Target Variations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

def prepare_farmer_data_variations():
    """Prepare farmer dataset with 3 different target combinations"""
    
    print("=== FARMER ADVISOR DATASET PREPARATION ===")
    
    # Load original dataset
    df = pd.read_csv('data/original_data/farmer_advisor_dataset.csv')
    print(f"Loaded farmer dataset: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create output directory
    os.makedirs('data/prepared', exist_ok=True)
    
    # ==========================================
    # VARIATION 1: YIELD & SUSTAINABILITY PREDICTION (2 targets)
    # ==========================================
    print("\n=== VARIATION 1: YIELD & SUSTAINABILITY PREDICTION ===")
    
    # Features: All columns except ID and targets
    yield_features = df[['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm',
                        'Fertilizer_Usage_kg', 'Pesticide_Usage_kg', 'Crop_Type']].copy()
    
    # Add small constant to avoid zeros in numeric features
    yield_numeric = yield_features[['Soil_pH', 'Soil_Moisture', 'Temperature_C', 
                                   'Rainfall_mm', 'Fertilizer_Usage_kg', 'Pesticide_Usage_kg']].copy()
    yield_numeric = yield_numeric + 0.001
    
    # One-hot encode Crop_Type for numeric features
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    crop_encoded = encoder.fit_transform(yield_features[['Crop_Type']])
    crop_names = encoder.get_feature_names_out(['Crop_Type'])
    
    # Add small constant to one-hot encoded features to avoid zeros
    crop_encoded = crop_encoded + 0.001
    
    # Combine numeric features with encoded categorical
    yield_combined = np.hstack([yield_numeric.values, crop_encoded])
    yield_feature_names = list(yield_numeric.columns) + list(crop_names)
    
    # Targets: Crop_Yield_ton and Sustainability_Score
    yield_targets = df[['Crop_Yield_ton', 'Sustainability_Score']].copy()
    
    # Add small constant to avoid zeros in targets
    yield_targets = yield_targets + 0.001
    
    # Rename targets with target_ prefix for GINN
    yield_targets = yield_targets.rename(columns={
        'Crop_Yield_ton': 'target_Crop_Yield_ton',
        'Sustainability_Score': 'target_Sustainability_Score'
    })
    
    # Create final dataset
    yield_prediction = pd.DataFrame(yield_combined, columns=yield_feature_names)
    yield_prediction['target_Crop_Yield_ton'] = yield_targets['target_Crop_Yield_ton'].values
    yield_prediction['target_Sustainability_Score'] = yield_targets['target_Sustainability_Score'].values
    
    # Save
    yield_prediction.to_csv('data/prepared/farmer_yield_sustainability.csv', index=False)
    print("✅ Saved: farmer_yield_sustainability.csv")
    print(f"   Features: {yield_feature_names}")
    print(f"   Targets: target_Crop_Yield_ton, target_Sustainability_Score")
    print(f"   Purpose: Predict crop yield and sustainability from farming conditions")
    
    # ==========================================
    # VARIATION 2: SOIL CONDITIONS PREDICTION (2 targets)
    # ==========================================
    print("\n=== VARIATION 2: SOIL CONDITIONS PREDICTION ===")
    
    # Features: All columns except ID and soil targets
    soil_features = df[['Temperature_C', 'Rainfall_mm', 'Fertilizer_Usage_kg', 
                       'Pesticide_Usage_kg', 'Crop_Type', 'Crop_Yield_ton', 
                       'Sustainability_Score']].copy()
    
    # Add small constant to avoid zeros in numeric features
    soil_numeric = soil_features[['Temperature_C', 'Rainfall_mm', 'Fertilizer_Usage_kg',
                                 'Crop_Yield_ton', 'Sustainability_Score']].copy()
    soil_numeric = soil_numeric + 0.001
    
    # One-hot encode Crop_Type
    soil_encoded = encoder.fit_transform(soil_features[['Crop_Type']])
    soil_names = encoder.get_feature_names_out(['Crop_Type'])
    
    # Add small constant to one-hot encoded features to avoid zeros
    soil_encoded = soil_encoded + 0.001
    
    # Combine features
    soil_combined = np.hstack([soil_numeric.values, soil_encoded])
    soil_feature_names = list(soil_numeric.columns) + list(soil_names)
    
    # Targets: Soil_pH and Soil_Moisture
    soil_targets = df[['Soil_pH', 'Soil_Moisture']].copy()
    
    # Add small constant to avoid zeros in targets
    soil_targets = soil_targets + 0.001
    
    # Rename targets with target_ prefix for GINN
    soil_targets = soil_targets.rename(columns={
        'Soil_pH': 'target_Soil_pH',
        'Soil_Moisture': 'target_Soil_Moisture'
    })
    
    # Create final dataset
    soil_prediction = pd.DataFrame(soil_combined, columns=soil_feature_names)
    soil_prediction['target_Soil_pH'] = soil_targets['target_Soil_pH'].values
    soil_prediction['target_Soil_Moisture'] = soil_targets['target_Soil_Moisture'].values
    
    # Save
    soil_prediction.to_csv('data/prepared/farmer_soil_conditions.csv', index=False)
    print("✅ Saved: farmer_soil_conditions.csv")
    print(f"   Features: {soil_feature_names}")
    print(f"   Targets: target_Soil_pH, target_Soil_Moisture")
    print(f"   Purpose: Predict soil conditions from farming practices and environment")
    
    # ==========================================
    # VARIATION 3: RESOURCE OPTIMIZATION PREDICTION (2 targets)
    # ==========================================
    print("\n=== VARIATION 3: RESOURCE OPTIMIZATION PREDICTION ===")
    
    # Features: All columns except ID and resource targets
    resource_features = df[['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm',
                           'Crop_Type', 'Crop_Yield_ton', 'Sustainability_Score']].copy()
    
    # Add small constant to avoid zeros in numeric features
    resource_numeric = resource_features[['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm',
                                         'Crop_Yield_ton', 'Sustainability_Score']].copy()
    resource_numeric = resource_numeric + 0.001
    
    # One-hot encode Crop_Type
    resource_encoded = encoder.fit_transform(resource_features[['Crop_Type']])
    resource_names = encoder.get_feature_names_out(['Crop_Type'])
    
    # Add small constant to one-hot encoded features to avoid zeros
    resource_encoded = resource_encoded + 0.001
    
    # Combine features
    resource_combined = np.hstack([resource_numeric.values, resource_encoded])
    resource_feature_names = list(resource_numeric.columns) + list(resource_names)
    
    # Targets: Fertilizer_Usage_kg and Pesticide_Usage_kg
    resource_targets = df[['Fertilizer_Usage_kg', 'Pesticide_Usage_kg']].copy()
    
    # Add small constant to avoid zeros in targets
    resource_targets = resource_targets + 0.001
    
    # Rename targets with target_ prefix for GINN
    resource_targets = resource_targets.rename(columns={
        'Fertilizer_Usage_kg': 'target_Fertilizer_Usage_kg',
        'Pesticide_Usage_kg': 'target_Pesticide_Usage_kg'
    })
    
    # Create final dataset
    resource_prediction = pd.DataFrame(resource_combined, columns=resource_feature_names)
    resource_prediction['target_Fertilizer_Usage_kg'] = resource_targets['target_Fertilizer_Usage_kg'].values
    resource_prediction['target_Pesticide_Usage_kg'] = resource_targets['target_Pesticide_Usage_kg'].values
    
    # Save
    resource_prediction.to_csv('data/prepared/farmer_resource_optimization.csv', index=False)
    print("✅ Saved: farmer_resource_optimization.csv")
    print(f"   Features: {resource_feature_names}")
    print(f"   Targets: target_Fertilizer_Usage_kg, target_Pesticide_Usage_kg")
    print(f"   Purpose: Predict optimal resource usage from farming conditions")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n=== SUMMARY ===")
    print("Created 3 farmer dataset variations optimized for GINN:")
    print("\n1. farmer_yield_sustainability.csv")
    print("   - Predicts: Crop yield and sustainability score")
    print("   - Features: Soil, weather, resources, crop type (one-hot encoded)")
    print("   - Targets: target_Crop_Yield_ton, target_Sustainability_Score")
    print("   - Use case: 'What yield and sustainability can I expect?'")
    
    print("\n2. farmer_soil_conditions.csv")
    print("   - Predicts: Soil pH and moisture levels")
    print("   - Features: Weather, resources, crop type, yield/sustainability")
    print("   - Targets: target_Soil_pH, target_Soil_Moisture")
    print("   - Use case: 'What will my soil conditions be?'")
    
    print("\n3. farmer_resource_optimization.csv")
    print("   - Predicts: Optimal fertilizer and pesticide usage")
    print("   - Features: Soil, weather, crop type, yield/sustainability")
    print("   - Targets: target_Fertilizer_Usage_kg, target_Pesticide_Usage_kg")
    print("   - Use case: 'How much fertilizer and pesticide should I use?'")
    
    print("\n✅ All 3 variations created successfully!")
    print("🎯 All targets are continuous and suitable for GINN regression!")
    print("🌾 Crop_Type is now a feature (one-hot encoded) in all variations!")
    
    return {
        'yield_sustainability': yield_prediction,
        'soil_conditions': soil_prediction,
        'resource_optimization': resource_prediction
    }

if __name__ == "__main__":
    prepare_farmer_data_variations()
