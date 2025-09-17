#!/usr/bin/env python3
"""
Quick script to test raw equations from POS dataset on test data.
This tests the equations without any refitting (no data leakage).
"""

import json
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def custom_mape(y_true, y_pred):
    """Custom MAPE calculation to match the original implementation."""
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    
    # Calculate MAPE only for non-zero true values
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def parse_equation(expr_str):
    """Parse the equation string and create a callable function."""
    # Clean up the expression - replace X_ with x_ for eval
    expr = expr_str.replace('X_', 'x_')
    
    # Create a function that evaluates the expression
    # The equations are complex polynomials with quadratic and interaction terms
    def evaluate(x):
        # x should be a list/array of 10 values
        x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10 = x
        return eval(expr)
    
    return evaluate

def main():
    # Load the JSON results
    with open('outputs/ginn_multi_POS.json', 'r') as f:
        results = json.load(f)
    
    # Get the first fold results
    fold_data = results[0]
    
    print("=== Testing Raw Equations on Test Data ===\n")
    print("IMPORTANT: We are using SCALED test data with SCALED equations")
    print("This matches the original evaluation (no data leakage)\n")
    
    for target_idx, target_data in enumerate(fold_data['per_target']):
        target_name = target_data['target']
        print(f"--- {target_name.upper()} ---")
        
        # Get the raw equation expression (SCALED version)
        raw_expr = target_data['expr']
        print(f"Raw Equation (scaled): {raw_expr[:100]}...")
        
        # Get test data (SCALED)
        X_test = np.array(fold_data['test_data']['X_test'])
        Y_test = np.array(fold_data['test_data']['Y_test'])
        
        print(f"Test data shape: X={X_test.shape}, Y={Y_test.shape}")
        print(f"First test sample X: {X_test[0]}")
        print(f"First test sample Y: {Y_test[0]}")
        
        # Parse and test the raw equation
        try:
            # Create evaluator for the polynomial
            equation_func = parse_equation(raw_expr)
            
            # Make predictions using the raw equation
            predictions = []
            for i, x_sample in enumerate(X_test):
                try:
                    pred = equation_func(x_sample)
                    predictions.append(pred)
                except Exception as e:
                    print(f"Error evaluating sample {i}: {e}")
                    predictions.append(0)  # fallback
            
            predictions = np.array(predictions)
            true_values = Y_test[:, target_idx]  # Get the target column
            
            # Calculate metrics
            mse = mean_squared_error(true_values, predictions)
            mae = mean_absolute_error(true_values, predictions)
            mape = custom_mape(true_values, predictions)
            r2 = r2_score(true_values, predictions)
            
            print(f"\nRaw Equation Performance (our calculation):")
            print(f"  R²: {r2:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            
            # Compare with JSON metrics
            json_metrics = target_data['eq']
            print(f"\nJSON Metrics (from original run):")
            print(f"  R²: {json_metrics['R2']:.4f}")
            print(f"  MSE: {json_metrics['MSE']:.4f}")
            print(f"  MAE: {json_metrics['MAE']:.4f}")
            print(f"  MAPE: {json_metrics['MAPE']:.2f}%")
            
            # Check if they match
            print(f"\nMetrics match?")
            print(f"  R²: {abs(r2 - json_metrics['R2']) < 0.001}")
            print(f"  MSE: {abs(mse - json_metrics['MSE']) < 0.001}")
            print(f"  MAE: {abs(mae - json_metrics['MAE']) < 0.001}")
            print(f"  MAPE: {abs(mape - json_metrics['MAPE']) < 0.01}")
            
            # Debug: Show first few predictions vs true values
            print(f"\nFirst 5 predictions vs true values:")
            for i in range(min(5, len(predictions))):
                print(f"  Sample {i}: Pred={predictions[i]:.4f}, True={true_values[i]:.4f}, Diff={abs(predictions[i]-true_values[i]):.4f}")
            
        except Exception as e:
            print(f"Error parsing/evaluating equation: {e}")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
