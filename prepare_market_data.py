#!/usr/bin/env python3
"""
Prepare Market Researcher Dataset for GINN
Targets: Market_Price_per_ton, Demand_Index, Weather_Impact_Score
Features: All other numeric columns + one-hot encoded categoricals
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import os

def prepare_market_data_for_ginn():
    """Prepare market dataset for GINN with proper feature/target separation"""
    
    # Load data
    df = pd.read_csv('data/original_data/market_researcher_dataset.csv')
    print(f"Loaded market dataset: {df.shape}")
    
    # Define targets (3 targets for GINN)
    target_cols = [
        'Market_Price_per_ton',
        'Demand_Index', 
        'Weather_Impact_Score'
    ]
    
    # Define feature columns (exclude ID and targets)
    feature_cols = [
        'Supply_Index',
        'Competitor_Price_per_ton', 
        'Economic_Indicator',
        'Consumer_Trend_Index',
        'Product',  # Categorical
        'Seasonal_Factor'  # Categorical
    ]
    
    print(f"Target columns: {target_cols}")
    print(f"Feature columns: {feature_cols}")
    
    # Extract features and targets
    X_categorical = df[['Product', 'Seasonal_Factor']].copy()
    X_numeric = df[['Supply_Index', 'Competitor_Price_per_ton', 'Economic_Indicator', 'Consumer_Trend_Index']].copy()
    y = df[target_cols].copy()
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_cat_encoded = encoder.fit_transform(X_categorical)
    cat_feature_names = encoder.get_feature_names_out(['Product', 'Seasonal_Factor'])
    
    # Combine numeric and encoded categorical features
    X_combined = np.hstack([X_numeric.values, X_cat_encoded])
    
    # Create feature names
    feature_names = list(X_numeric.columns) + list(cat_feature_names)
    
    print(f"Final feature matrix shape: {X_combined.shape}")
    print(f"Target matrix shape: {y.shape}")
    print(f"Feature names: {feature_names}")
    print(f"Target names: {target_cols}")
    
    # Scale features (GINN requirement)
    scaler = MinMaxScaler(feature_range=(0.1, 1.0))  # Avoid zeros for GINN
    X_scaled = scaler.fit_transform(X_combined)
    
    # Keep targets unscaled (GINN requirement)
    y_values = y.values.astype(np.float32)
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_values, test_size=0.2, random_state=42
    )
    
    print(f"Training set: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set: X={X_test.shape}, y={y_test.shape}")
    
    # Save prepared data
    os.makedirs('data/prepared', exist_ok=True)
    
    # Save as CSV for GINN
    train_df = pd.DataFrame(X_train, columns=feature_names)
    for i, target in enumerate(target_cols):
        train_df[f'target_{i+1}'] = y_train[:, i]
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    for i, target in enumerate(target_cols):
        test_df[f'target_{i+1}'] = y_test[:, i]
    
    train_df.to_csv('data/prepared/market_train_ginn.csv', index=False)
    test_df.to_csv('data/prepared/market_test_ginn.csv', index=False)
    
    print("✅ Market data prepared and saved!")
    print("   - data/prepared/market_train_ginn.csv")
    print("   - data/prepared/market_test_ginn.csv")
    
    return X_scaled, y_values, feature_names, target_cols

if __name__ == "__main__":
    prepare_market_data_for_ginn()
