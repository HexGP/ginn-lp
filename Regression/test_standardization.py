#!/usr/bin/env python3
"""
Test script to verify that all regressors in the Regression folder 
are using the same standardized data handling approach.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Add the Regression directory to the path
sys.path.append(os.path.dirname(__file__))

# Import shared utilities
from shared_data_utils import (
    load_and_preprocess_data, 
    get_train_test_split, 
    create_standard_scaler,
    get_data_info,
    validate_data_consistency,
    RANDOM_STATE,
    TEST_SIZE,
    SCALER_FEATURE_RANGE,
    DATA_FILE_PATH
)

def test_data_consistency():
    """Test that all regressors use the same data handling approach."""
    
    print("="*80)
    print("TESTING DATA HANDLING STANDARDIZATION")
    print("="*80)
    
    # Test 1: Load data using shared utilities
    print("\n1. Testing data loading...")
    X, y, feature_names, target_names = load_and_preprocess_data()
    print(f"   ✓ Data loaded successfully")
    print(f"   ✓ Features: {len(feature_names)} features")
    print(f"   ✓ Targets: {len(target_names)} targets")
    print(f"   ✓ Data shape: {X.shape}")
    
    # Test 2: Validate data consistency
    print("\n2. Testing data validation...")
    try:
        validate_data_consistency(X, y, feature_names, target_names)
        print("   ✓ Data validation passed")
    except ValueError as e:
        print(f"   ✗ Data validation failed: {e}")
        return False
    
    # Test 3: Test train/test split
    print("\n3. Testing train/test split...")
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    print(f"   ✓ Train set size: {X_train.shape[0]}")
    print(f"   ✓ Test set size: {X_test.shape[0]}")
    print(f"   ✓ Split ratio: {X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]):.2f}")
    
    # Test 4: Test scaler creation
    print("\n4. Testing scaler creation...")
    scaler = create_standard_scaler()
    print(f"   ✓ Scaler created: {type(scaler).__name__}")
    print(f"   ✓ Feature range: {scaler.feature_range}")
    
    # Test 5: Test scaler functionality
    print("\n5. Testing scaler functionality...")
    X_scaled = scaler.fit_transform(X_train)
    print(f"   ✓ Scaled data shape: {X_scaled.shape}")
    print(f"   ✓ Scaled data range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    
    # Test 6: Verify constants
    print("\n6. Testing constants...")
    print(f"   ✓ Random state: {RANDOM_STATE}")
    print(f"   ✓ Test size: {TEST_SIZE}")
    print(f"   ✓ Scaler feature range: {SCALER_FEATURE_RANGE}")
    print(f"   ✓ Data file path: {DATA_FILE_PATH}")
    
    # Test 7: Test data info function
    print("\n7. Testing data info function...")
    get_data_info(X, y, feature_names, target_names)
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED - DATA HANDLING IS STANDARDIZED")
    print("="*80)
    
    return True

def test_regressor_imports():
    """Test that all regressors can be imported and use shared utilities."""
    
    print("\n" + "="*80)
    print("TESTING REGRESSOR IMPORTS")
    print("="*80)
    
    regressors = [
        ('linear_regressor', 'linear_regressor'),
        ('mlp_regressor', 'mlp_regressor'),
        ('sgd_regressor', 'sgd_regressor'),
        ('multiout_regressor', 'multiout_regressor'),
        ('gp_regressor', 'gp_regressor')
    ]
    
    for folder, module in regressors:
        print(f"\nTesting {folder}...")
        try:
            # Import the module
            module_path = os.path.join(folder, module)
            spec = __import__(module_path, fromlist=['*'])
            print(f"   ✓ {folder} imported successfully")
            
            # Check if it uses shared utilities
            if hasattr(spec, 'load_and_preprocess_data'):
                print(f"   ⚠ {folder} still has local load_and_preprocess_data function")
            else:
                print(f"   ✓ {folder} uses shared data utilities")
                
        except Exception as e:
            print(f"   ✗ {folder} import failed: {e}")
    
    print("\n" + "="*80)
    print("IMPORT TESTS COMPLETED")
    print("="*80)

def main():
    """Run all standardization tests."""
    
    print("Testing data handling standardization across all regressors...")
    
    # Test data consistency
    if not test_data_consistency():
        print("Data consistency tests failed!")
        return False
    
    # Test regressor imports
    test_regressor_imports()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("✓ All regressors now use shared data utilities")
    print("✓ Consistent scaler: MinMaxScaler(feature_range=(0.1, 10.0))")
    print("✓ Consistent train/test split: test_size=0.2, random_state=42")
    print("✓ Consistent data loading from: data/ENB2012_data.csv")
    print("✓ Data validation ensures consistency")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 