import pandas as pd
import numpy as np
import sys
import os
import warnings
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import shared utilities
from data_utils_ENB import (
    load_and_preprocess_data, 
    get_train_test_split, 
    create_standard_scaler,
    get_data_info,
    validate_data_consistency
)

# Import all regressor modules
from ENB_Regression.linear_regressor.linear_regressor import MultiTaskLinearRegression
from ENB_Regression.gp_regressor.gp_regressor import MultiTaskGaussianProcess
from ENB_Regression.mlp_regressor.mlp_regressor import MLPRegressionModel
from ENB_Regression.multiout_regressor.multiout_regressor import MultiOutputRegressionModel
from ENB_Regression.sgd_regressor.sgd_regressor import MultiTaskSGDRegressor

def run_linear_regressor(X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray,
                        feature_names: List[str], target_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Run Linear Regressor and return results."""
    print("Running Linear Regressor...")
    
    model = MultiTaskLinearRegression(random_state=42)
    model.fit(X_train, y_train, feature_names, target_names)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Evaluate directly (no scaling needed since we're using smoothing)
    results = {}
    for i, target_name in enumerate(target_names):
        mse = np.mean((y_test[:, i] - y_pred[:, i])**2)
        mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / y_test[:, i])) * 100
        
        results[target_name] = {
            'MSE': mse,
            'MAPE': mape
        }
    
    return results

def run_gp_regressor(X_train: np.ndarray, y_train: np.ndarray, 
                     X_test: np.ndarray, y_test: np.ndarray,
                     feature_names: List[str], target_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Run GP Regressor and return results."""
    print("Running GP Regressor...")
    
    model = MultiTaskGaussianProcess(random_state=42)
    model.fit(X_train, y_train, feature_names, target_names)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Evaluate directly (no scaling needed since we're using smoothing)
    results = {}
    for i, target_name in enumerate(target_names):
        mse = np.mean((y_test[:, i] - y_pred[:, i])**2)
        mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / y_test[:, i])) * 100
        
        results[target_name] = {
            'MSE': mse,
            'MAPE': mape
        }
    
    return results

def run_mlp_regressor(X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray,
                      feature_names: List[str], target_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Run MLP Regressor and return results."""
    print("Running MLP Regressor...")
    
    model = MLPRegressionModel(random_state=42)
    model.fit(X_train, y_train, feature_names, target_names)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Evaluate directly (no scaling needed since we're using smoothing)
    results = {}
    for i, target_name in enumerate(target_names):
        mse = np.mean((y_test[:, i] - y_pred[:, i])**2)
        mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / y_test[:, i])) * 100
        
        results[target_name] = {
            'MSE': mse,
            'MAPE': mape
        }
    
    return results

def run_multiout_regressor(X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray,
                          feature_names: List[str], target_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Run MultiOutput Regressor and return results."""
    print("Running MultiOutput Regressor...")
    
    model = MultiOutputRegressionModel(random_state=42)
    model.fit(X_train, y_train, feature_names, target_names)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Evaluate directly (no scaling needed since we're using smoothing)
    results = {}
    for i, target_name in enumerate(target_names):
        mse = np.mean((y_test[:, i] - y_pred[:, i])**2)
        mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / y_test[:, i])) * 100
        
        results[target_name] = {
            'MSE': mse,
            'MAPE': mape
        }
    
    return results

def run_sgd_regressor(X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray,
                      feature_names: List[str], target_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Run SGD Regressor and return results."""
    print("Running SGD Regressor...")
    
    model = MultiTaskSGDRegressor(random_state=42)
    model.fit(X_train, y_train, feature_names, target_names)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Evaluate directly (no scaling needed since we're using smoothing)
    results = {}
    for i, target_name in enumerate(target_names):
        mse = np.mean((y_test[:, i] - y_pred[:, i])**2)
        mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / y_test[:, i])) * 100
        
        results[target_name] = {
            'MSE': mse,
            'MAPE': mape
        }
    
    return results

def create_comparison_table(all_results: Dict[str, Dict[str, Dict[str, float]]], 
                           target_names: List[str]) -> pd.DataFrame:
    """Create the comparison table in the specified format."""
    
    # Define model names in the order they should appear
    model_names = [
        'Linear Regressor',
        'GP Regressor', 
        'MLP Regressor',
        'MultiOutput Regressor',
        'SGD Regressor'
    ]
    
    # Create the table structure
    table_data = []
    
    for model_name in model_names:
        if model_name in all_results:
            results = all_results[model_name]
            
            # Extract MSE and MAPE for each target
            y1_mse = results[target_names[0]]['MSE']
            y2_mse = results[target_names[1]]['MSE']
            avg_mse = (y1_mse + y2_mse) / 2
            
            y1_mape = results[target_names[0]]['MAPE']
            y2_mape = results[target_names[1]]['MAPE']
            avg_mape = (y1_mape + y2_mape) / 2
            
            table_data.append({
                'Model': model_name,
                'Y1_MSE': y1_mse,
                'Y2_MSE': y2_mse,
                'Avg_MSE': avg_mse,
                'Y1_MAPE': y1_mape,
                'Y2_MAPE': y2_mape,
                'Avg_MAPE': avg_mape
            })
    
    return pd.DataFrame(table_data)

def print_formatted_table(df: pd.DataFrame):
    """Print the table in the exact format specified."""
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Print header
    print(f"{'Model':<25} | {'MSE':<15} | {'MAPE':<15}")
    print(f"{'':<25} | {'Y1':<6} {'Y2':<6} {'Avg':<6} | {'Y1':<6} {'Y2':<6} {'Avg':<6}")
    print("-" * 80)
    
    # Print data rows
    for _, row in df.iterrows():
        print(f"{row['Model']:<25} | {row['Y1_MSE']:<6.4f} {row['Y2_MSE']:<6.4f} {row['Avg_MSE']:<6.4f} | "
              f"{row['Y1_MAPE']:<6.2f} {row['Y2_MAPE']:<6.2f} {row['Avg_MAPE']:<6.2f}")
    
    print("="*80)

def save_detailed_results_to_json(all_results, comparison_df, target_names, output_dir=None):
    """Save all results including smoothing details to JSON."""
    import json
    
    # Create comprehensive results dictionary
    json_results = {
        "dataset_info": {
            "name": "ENB2012_data.csv",
            "preprocessing": "Savgol smoothing + positivity clamp",
            "window_length": 15,
            "polynomial_order": 3,
            "min_positive": 0.01
        },
        "model_results": all_results,
        "comparison_table": comparison_df.to_dict('records'),
        "target_names": target_names
    }
    
    # Save to JSON
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = script_dir
    
    json_file = os.path.join(output_dir, 'comparison_results_ENB.json')
    
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"Detailed results saved to '{json_file}'")
    return json_file

def main():
    """Main function to run all models and generate comparison table."""
    
    print("Loading and preprocessing dataset with smoothing approach (matching GINN preprocessing)...")
    
    # Load data using shared utilities
    X, y, feature_names, target_names = load_and_preprocess_data()
    
    # Validate data consistency
    validate_data_consistency(X, y, feature_names, target_names)
    
    # Print standardized data information
    get_data_info(X, y, feature_names, target_names)
    
    # Split data using shared utilities
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Features: {feature_names}")
    print(f"Targets: {target_names}")
    
    # Dictionary to store all results
    all_results = {}
    
    # Run all models
    print("\n" + "="*80)
    print("RUNNING ALL MODELS")
    print("="*80)
    
    try:
        # Linear Regressor
        all_results['Linear Regressor'] = run_linear_regressor(
            X_train, y_train, X_test, y_test, feature_names, target_names
        )
    except Exception as e:
        print(f"Error running Linear Regressor: {e}")
    
    try:
        # GP Regressor
        all_results['GP Regressor'] = run_gp_regressor(
            X_train, y_train, X_test, y_test, feature_names, target_names
        )
    except Exception as e:
        print(f"Error running GP Regressor: {e}")
    
    try:
        # MLP Regressor
        all_results['MLP Regressor'] = run_mlp_regressor(
            X_train, y_train, X_test, y_test, feature_names, target_names
        )
    except Exception as e:
        print(f"Error running MLP Regressor: {e}")
    
    try:
        # MultiOutput Regressor
        all_results['MultiOutput Regressor'] = run_multiout_regressor(
            X_train, y_train, X_test, y_test, feature_names, target_names
        )
    except Exception as e:
        print(f"Error running MultiOutput Regressor: {e}")
    
    try:
        # SGD Regressor
        all_results['SGD Regressor'] = run_sgd_regressor(
            X_train, y_train, X_test, y_test, feature_names, target_names
        )
    except Exception as e:
        print(f"Error running SGD Regressor: {e}")
    
    # Create comparison table
    print("\n" + "="*80)
    print("GENERATING COMPARISON TABLE")
    print("="*80)
    
    comparison_df = create_comparison_table(all_results, target_names)
    
    # Print formatted table
    print_formatted_table(comparison_df)
    
    # Save results to CSV in all_outputs folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'all_outputs', 'ENB')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'comparison_ENB.csv')
    comparison_df.to_csv(output_file, index=False)
    print(f"\nResults saved to '{output_file}'")
    
    # Also save detailed results
    detailed_results = {}
    for model_name, results in all_results.items():
        detailed_results[model_name] = {}
        for target_name, metrics in results.items():
            detailed_results[model_name][f"{target_name}_MSE"] = metrics['MSE']
            detailed_results[model_name][f"{target_name}_MAPE"] = metrics['MAPE']
    
    detailed_df = pd.DataFrame(detailed_results).T
    detailed_output_file = os.path.join(output_dir, 'comparison_detail_ENB.csv')
    detailed_df.to_csv(detailed_output_file)
    print(f"Detailed results saved to '{detailed_output_file}'")
    
    # Save detailed results to JSON
    save_detailed_results_to_json(all_results, comparison_df, target_names, output_dir)
    
    return comparison_df, all_results

if __name__ == "__main__":
    comparison_df, all_results = main() 