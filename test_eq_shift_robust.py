#!/usr/bin/env python3
"""
Robustness test script to validate equations with different preprocessing shifts
Tests: +0.05 (training), +0.5 (10x larger), -0.05 (opposite direction)
"""

import json
import numpy as np
import pandas as pd
import random
from sympy import symbols, lambdify

def load_equations_from_json(json_file):
    """Load equations from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get the first fold's equations
    fold_data = data[0]
    equations = {}
    
    for target_data in fold_data['per_target']:
        target_name = target_data['target']
        equations[target_name] = {
            'raw': target_data['expr'],
            'refit': target_data['expr_refit']
        }
    
    return equations

def preprocess_features(X, shift=0.05):
    """Apply shift to features"""
    return X + shift

def evaluate_equation(equation_str, X_values):
    """Evaluate equation string with given X values"""
    # Create symbols for X_1 to X_8
    X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8 = symbols('X_1 X_2 X_3 X_4 X_5 X_6 X_7 X_8')
    
    # Parse and compile the equation
    try:
        # Evaluate the equation string in the local namespace with symbols
        local_vars = {'X_1': X_1, 'X_2': X_2, 'X_3': X_3, 'X_4': X_4, 
                     'X_5': X_5, 'X_6': X_6, 'X_7': X_7, 'X_8': X_8}
        expr = eval(equation_str, {"__builtins__": {}}, local_vars)
        
        # Create lambda function
        f = lambdify([X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8], expr, 'numpy')
        
        # Evaluate with X values
        result = f(*X_values[0])  # X_values is 2D, take first row
        return result
    except Exception as e:
        print(f"Error evaluating equation: {e}")
        return None

def calculate_metrics(y_true, y_pred):
    """Calculate various regression metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def test_robustness(csv_file, json_file, num_samples=5):
    """Test equations with different preprocessing shifts"""
    
    # Load data
    print("📊 Loading ENB dataset...")
    df = pd.read_csv(csv_file)
    
    # Separate features and targets
    X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']].values
    y = df[['target_1', 'target_2']].values
    
    print(f"   Dataset shape: {X.shape}, targets: {y.shape}")
    
    # Load equations
    print("🔧 Loading equations from JSON...")
    equations = load_equations_from_json(json_file)
    
    # Define test scenarios
    test_scenarios = [
        {"name": "Training (+0.05)", "shift": 0.05, "description": "Same as training preprocessing"},
        {"name": "Large (+0.5)", "shift": 0.5, "description": "10x larger shift - stress test"},
        {"name": "Opposite (-0.05)", "shift": -0.05, "description": "Opposite direction - robustness test"},
        {"name": "Large Opposite (-0.5)", "shift": -0.5, "description": "10x opposite shift - theory test"}
    ]
    
    # Pick random samples (same for all tests)
    random.seed(42)  # Same samples for fair comparison
    sample_indices = [random.randint(0, len(X) - 1) for _ in range(num_samples)]
    
    print(f"\n🎯 Testing on {num_samples} random samples (indices: {sample_indices})")
    print("="*80)
    
    # Test each scenario
    for scenario in test_scenarios:
        print(f"\n🧪 SCENARIO: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Shift value: {scenario['shift']}")
        print("-" * 60)
        
    # Store results for metrics calculation
    all_true_values = {'target_1': [], 'target_2': []}
    all_raw_predictions = {'target_1': [], 'target_2': []}
    all_refit_predictions = {'target_1': [], 'target_2': []}
    
    # Store results for comprehensive tables
    results = {}
    sample_results = []
    
    # Test each scenario
    for scenario in test_scenarios:
        print(f"\n🧪 SCENARIO: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Shift value: {scenario['shift']}")
        print("-" * 60)
        
        # Store results for metrics calculation
        all_true_values = {'target_1': [], 'target_2': []}
        all_raw_predictions = {'target_1': [], 'target_2': []}
        all_refit_predictions = {'target_1': [], 'target_2': []}
        
        # Test each sample
        for i, idx in enumerate(sample_indices):
            X_sample = X[idx:idx+1]  # Keep as 2D
            y_sample = y[idx]
            
            print(f"\n--- Sample {i+1} (Index {idx}) ---")
            print(f"Original features: {X_sample[0]}")
            print(f"True targets: {y_sample}")
            
            # Apply preprocessing with current shift
            X_shifted = preprocess_features(X_sample, shift=scenario['shift'])
            print(f"Shifted features: {X_shifted[0]}")
            
            # Test each equation
            for target_name, eqs in equations.items():
                target_idx = 0 if target_name == 'target_1' else 1
                true_value = y_sample[target_idx]
                
                print(f"\n{target_name} (True: {true_value:.3f}):")
                
                # Test raw equation
                raw_pred = evaluate_equation(eqs['raw'], X_shifted)
                if raw_pred is not None:
                    raw_val = raw_pred[0] if hasattr(raw_pred, '__len__') else raw_pred
                    raw_error = abs(raw_val - true_value)
                    print(f"  Raw prediction: {raw_val:.3f} (Error: {raw_error:.3f})")
                    
                    # Store for metrics
                    all_true_values[target_name].append(true_value)
                    all_raw_predictions[target_name].append(raw_val)
                
                # Test refit equation
                refit_pred = evaluate_equation(eqs['refit'], X_shifted)
                if refit_pred is not None:
                    refit_val = refit_pred[0] if hasattr(refit_pred, '__len__') else refit_pred
                    refit_error = abs(refit_val - true_value)
                    print(f"  Refit prediction: {refit_val:.3f} (Error: {refit_error:.3f})")
                    
                    # Store for metrics
                    all_refit_predictions[target_name].append(refit_val)
        
        # Calculate and display metrics for this scenario
        print(f"\n📊 METRICS FOR {scenario['name'].upper()}:")
        print("-" * 50)
        
        for target_name in ['target_1', 'target_2']:
            print(f"\n🎯 {target_name.upper()}:")
            
            # Calculate metrics for raw equations
            raw_metrics = calculate_metrics(all_true_values[target_name], all_raw_predictions[target_name])
            
            # Calculate metrics for refit equations
            refit_metrics = calculate_metrics(all_true_values[target_name], all_refit_predictions[target_name])
            
            # Display comprehensive metrics
            print(f"  Raw  - R²: {raw_metrics['R2']:.4f}, MAE: {raw_metrics['MAE']:.4f}, MAPE: {raw_metrics['MAPE']:.1f}%, MSE: {raw_metrics['MSE']:.4f}")
            print(f"  Refit - R²: {refit_metrics['R2']:.4f}, MAE: {refit_metrics['MAE']:.4f}, MAPE: {refit_metrics['MAPE']:.1f}%, MSE: {refit_metrics['MSE']:.4f}")
            
            # Calculate improvement
            r2_improvement = refit_metrics['R2'] - raw_metrics['R2']
            mae_improvement = raw_metrics['MAE'] - refit_metrics['MAE']
            mse_improvement = raw_metrics['MSE'] - refit_metrics['MSE']
            mape_improvement = raw_metrics['MAPE'] - refit_metrics['MAPE']
            print(f"  Improvement - R²: +{r2_improvement:.4f}, MAE: +{mae_improvement:.4f}, MAPE: +{mape_improvement:.1f}%, MSE: +{mse_improvement:.4f}")
    
    # Final summary
    print("\n" + "="*80)
    print("🎯 ROBUSTNESS TEST SUMMARY")
    print("="*80)
    print("This shows how sensitive your equations are to preprocessing changes:")
    print()
    print("✅ Training (+0.05): Should work well (baseline)")
    print("⚠️  Large (+0.5): Tests sensitivity to wrong preprocessing")
    print("⚠️  Opposite (-0.05): Tests sensitivity to direction")
    print()
    print("💡 INTERPRETATION:")
    print("   • If refit equations degrade gracefully → Robust approach")
    print("   • If refit equations break completely → Brittle approach")
    print("   • Raw equations should always be terrible (wrong scale)")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Test the equations with different preprocessing
    test_robustness('data/ENB2012_data.csv', 'outputs/JSON_ENB_shifted/grad_ENB_1S_6B_6L_shifted.json')
