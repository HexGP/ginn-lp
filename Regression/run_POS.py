import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import data utilities for POS dataset
from data_utils import get_data_utils
data_utils = get_data_utils('POS')
load_and_preprocess_data = data_utils['load_and_preprocess_data']
get_train_test_split = data_utils['get_train_test_split']
create_standard_scaler = data_utils['create_standard_scaler']
get_data_info = data_utils['get_data_info']
validate_data_consistency = data_utils['validate_data_consistency']

# Import individual regressors
from POS_Regression.linear_regressor.linear_regressor import MultiTaskLinearRegression
from POS_Regression.mlp_regressor.mlp_regressor import MLPRegressionModel
from POS_Regression.gp_regressor.gp_regressor import MultiTaskGaussianProcess
from POS_Regression.multiout_regressor.multiout_regressor import MultiOutputRegressionModel
from POS_Regression.sgd_regressor.sgd_regressor import MultiTaskSGDRegressor

def run_single_model(model_name: str, model_class, X_train: np.ndarray, X_test: np.ndarray, 
                    y_train: np.ndarray, y_test: np.ndarray, target_names: List[str]) -> Dict:
    """Run a single model and return results."""
    print(f"\n{'='*60}")
    print(f"Running {model_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize model
        if model_name == "Linear Regressor":
            model = model_class()
        elif model_name == "MLP Regressor":
            model = model_class(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        elif model_name == "GP Regressor":
            model = model_class()
        elif model_name == "MultiOutput Regressor":
            model = model_class()
        elif model_name == "SGD Regressor":
            model = model_class()
        else:
            model = model_class()
        
        # Train model
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        print(f"Making predictions with {model_name}...")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
        
        results = {}
        for i, target in enumerate(target_names):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            mape = mean_absolute_percentage_error(y_test[:, i], y_pred[:, i])
            
            results[f'{target}_MSE'] = mse
            results[f'{target}_MAPE'] = mape
            
            print(f"{target}: MSE = {mse:.4f}, MAPE = {mape:.2f}%")
        
        # Calculate averages
        mse_values = [results[f'{target}_MSE'] for target in target_names]
        mape_values = [results[f'{target}_MAPE'] for target in target_names]
        
        results['Avg_MSE'] = np.mean(mse_values)
        results['Avg_MAPE'] = np.mean(mape_values)
        
        print(f"Average MSE: {results['Avg_MSE']:.4f}")
        print(f"Average MAPE: {results['Avg_MAPE']:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"Error running {model_name}: {str(e)}")
        return {f'{target}_MSE': float('inf') for target in target_names} | \
               {f'{target}_MAPE': float('inf') for target in target_names} | \
               {'Avg_MSE': float('inf'), 'Avg_MAPE': float('inf')}

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

def save_detailed_results_to_json(all_results: List[Dict], comparison_df: pd.DataFrame, target_names: List[str], output_dir=None):
    """Save detailed results to JSON file."""
    import json
    
    # Create detailed results dictionary
    detailed_results = {
        'dataset_info': {
            'name': 'POS',
            'preprocessing': 'MinMaxScaler to range (0.1, 10)',
            'scaling_parameters': {
                'feature_range': (0.1, 10),
                'min_positive': 1e-2
            }
        },
        'target_names': target_names,
        'model_results': all_results,
        'comparison_table': comparison_df.to_dict('records')
    }
    
    # Save to JSON
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = script_dir
    
    json_output_file = os.path.join(output_dir, 'comparison_result_POS.json')
    
    with open(json_output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"Detailed results saved to '{json_output_file}'")

def main():
    """Main function to run all models and generate comparison."""
    print("POS Dataset Model Comparison")
    print("="*50)
    
    # Load and preprocess data
    print("Loading and preprocessing POS dataset...")
    X, y, feature_names, target_names = load_and_preprocess_data()
    
    # Validate data consistency
    validate_data_consistency(X, y, feature_names, target_names)
    
    # Get data info
    data_info = get_data_info(X, y, feature_names, target_names)
    print(f"\nDataset Info:")
    print(f"  Features: {data_info['num_features']} ({', '.join(feature_names)})")
    print(f"  Targets: {data_info['num_targets']} ({', '.join(target_names)})")
    print(f"  Samples: {data_info['num_samples']}")
    print(f"  Feature range: {data_info['feature_range']}")
    print(f"  Target range: {data_info['target_range']}")
    
    # Split data
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    print(f"\nData split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    # Define models to run
    models = [
        ("Linear Regressor", MultiTaskLinearRegression),
        ("MLP Regressor", MLPRegressionModel),
        ("GP Regressor", MultiTaskGaussianProcess),
        ("MultiOutput Regressor", MultiOutputRegressionModel),
        ("SGD Regressor", MultiTaskSGDRegressor)
    ]
    
    # Run all models
    all_results = []
    for model_name, model_class in models:
        results = run_single_model(model_name, model_class, X_train, X_test, y_train, y_test, target_names)
        results['Model'] = model_name
        all_results.append(results)
    
    # Create comparison DataFrame
    comparison_data = []
    for result in all_results:
        # Get the first two targets (assuming we have target_0 and target_1)
        target_0_mse = result.get('target_0_MSE', float('inf'))
        target_1_mse = result.get('target_1_MSE', float('inf'))
        target_0_mape = result.get('target_0_MAPE', float('inf'))
        target_1_mape = result.get('target_1_MAPE', float('inf'))
        
        row = {
            'Model': result['Model'],
            'Y1_MSE': target_0_mse,
            'Y2_MSE': target_1_mse,
            'Avg_MSE': result.get('Avg_MSE', float('inf')),
            'Y1_MAPE': target_0_mape,
            'Y2_MAPE': target_1_mape,
            'Avg_MAPE': result.get('Avg_MAPE', float('inf'))
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print formatted table
    print_formatted_table(comparison_df)
    
    # Save results to CSV in all_outputs folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'all_outputs', 'POS')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'comparison_POS.csv')
    comparison_df.to_csv(output_file, index=False)
    print(f"\nResults saved to '{output_file}'")
    
    # Save detailed results to JSON
    save_detailed_results_to_json(all_results, comparison_df, target_names, output_dir)
    
    return comparison_df, all_results

if __name__ == "__main__":
    comparison_df, all_results = main()
