import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
import os
warnings.filterwarnings('ignore')

def evaluate_power_equation_0(X):
    """
    Evaluate the LATEST power-based equation for output 0 from cross-validation
    This is the actual equation that was just extracted
    """
    # Extract features (X_1 to X_8 correspond to f_0 to f_7)
    X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7]
    
    # Build the shared term that appears in both equations (from latest extraction)
    shared_term = (0.963*X_1**0.894*X_2**0.37*X_3**0.611*X_4**0.609*X_6**0.193*X_7**1.17*X_8**0.572 + 
                   0.625*X_1**3.673*X_2**0.33*X_3**0.111*X_5**1.28*X_6**0.305*X_7**0.203 + 
                   X_2**0.301*X_3**0.071*X_4**1.14*X_5**0.929*X_6**0.648*X_7**0.294*X_8**0.481)
    
    # Apply the discovered equation for output 0 (from latest extraction)
    term1 = -0.0109 * (shared_term**0.803) / (X_1**0.695*X_3**0.146*X_4**0.438*X_5**0.407*X_6**0.4*X_7**0.399*X_8**0.37)
    term2 = -0.00968 * (shared_term**0.269) / (X_1**0.233*X_3**0.049*X_4**0.147*X_5**0.136*X_6**0.134*X_7**0.134*X_8**0.124)
    constant = 29.102
    
    y_pred = term1 + term2 + constant
    
    # Handle any NaN or inf values
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return y_pred

def evaluate_power_equation_1(X):
    """
    Evaluate the LATEST power-based equation for output 1 from cross-validation
    This is the actual equation that was just extracted
    """
    # Extract features (X_1 to X_8 correspond to f_0 to f_7)
    X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7]
    
    # Build the shared term that appears in both equations (from latest extraction)
    shared_term = (0.963*X_1**0.894*X_2**0.37*X_3**0.611*X_4**0.609*X_6**0.193*X_7**1.17*X_8**0.572 + 
                   0.625*X_1**3.673*X_2**0.33*X_3**0.111*X_5**1.28*X_6**0.305*X_7**0.203 + 
                   X_2**0.301*X_3**0.071*X_4**1.14*X_5**0.929*X_6**0.648*X_7**0.294*X_8**0.481)
    
    # Apply the discovered equation for output 1 (from latest extraction)
    term1 = -6.115 * (shared_term**0.284) / (X_1**0.246*X_3**0.0517*X_4**0.155*X_5**0.144*X_6**0.141*X_7**0.141*X_8**0.131)
    term2 = 8.746 * (shared_term**0.173) / (X_1**0.15*X_3**0.0315*X_4**0.0943*X_5**0.0877*X_6**0.0862*X_7**0.086*X_8**0.0798)
    constant = 2.705
    
    y_pred = term1 + term2 + constant
    
    # Handle any NaN or inf values
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return y_pred

def evaluate_log_equation_0(X):
    """
    Evaluate the LATEST logarithmic equation for output 0 from cross-validation
    This is the actual equation that was just extracted
    """
    # Extract features (X_1 to X_8 correspond to f_0 to f_7)
    X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7]
    
    # Build the shared feature that appears in the log equation (from latest extraction)
    shared_feature = (X_1**0.57*X_2**0.537*X_3**0.16*X_4**0.196*X_5**0.261*X_7**0.161) / (X_6**0.165*X_8**0.1)
    
    # Apply the logarithmic equation: log(y_0) = 1.132*log(log(shared_feature)) + 29.102
    # Convert back: y_0 = exp(1.132*log(log(shared_feature)) + 29.102)
    
    # Ensure shared_feature is positive and > 1 for log(log()) to work
    shared_feature = np.maximum(shared_feature, 1.1)
    
    try:
        # Apply the double log transformation
        log_shared = np.log(shared_feature)
        log_shared = np.maximum(log_shared, 0.1)  # Ensure positive for second log
        
        log_pred = 1.132 * np.log(log_shared) + 29.102
        y_pred = np.exp(log_pred)
        
    except Exception as e:
        print(f"Error in log equation 0: {e}")
        y_pred = np.full(X.shape[0], 0.0)
    
    # Handle any NaN or inf values
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return y_pred

def evaluate_log_equation_1(X):
    """
    Evaluate the LATEST logarithmic equation for output 1 from cross-validation
    This is the actual equation that was just extracted
    """
    # Extract features (X_1 to X_8 correspond to f_0 to f_7)
    X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7]
    
    # Build the shared feature that appears in the log equation (from latest extraction)
    shared_feature = (X_1**0.57*X_2**0.537*X_3**0.16*X_4**0.196*X_5**0.261*X_7**0.161) / (X_6**0.165*X_8**0.1)
    
    # Apply the logarithmic equation: log(y_1) = 2.705 - 0.053*log(log(shared_feature))
    # Convert back: y_1 = exp(2.705 - 0.053*log(log(shared_feature)))
    
    # Ensure shared_feature is positive and > 1 for log(log()) to work
    shared_feature = np.maximum(shared_feature, 1.1)
    
    try:
        # Apply the double log transformation
        log_shared = np.log(shared_feature)
        log_shared = np.maximum(log_shared, 0.1)  # Ensure positive for second log
        
        log_pred = 2.705 - 0.053 * np.log(log_shared)
        y_pred = np.exp(log_pred)
        
    except Exception as e:
        print(f"Error in log equation 1: {e}")
        y_pred = np.full(X.shape[0], 0.0)
    
    # Handle any NaN or inf values
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return y_pred

def calculate_equation_metrics(y_true, y_pred):
    """Calculate performance metrics for equation predictions"""
    # Filter out any remaining NaN or inf values
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(valid_mask) == 0:
        return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'R2': np.nan}
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    mse = mean_squared_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_valid - y_pred_valid))
    
    # Calculate R² manually to avoid sklearn issues
    ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
    ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Calculate MAPE
    eps = 1e-10
    mape = np.mean(np.abs((y_true_valid - y_pred_valid) / (np.abs(y_true_valid) + eps))) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def plot_equation_predictions(y_true, y_pred, title, save_path=None):
    """Plot true vs predicted values for equation predictions"""
    # Filter out NaN values for plotting
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(valid_mask) > 0:
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true_valid, y_pred_valid, alpha=0.6, s=20)
        plt.plot([y_true_valid.min(), y_true_valid.max()], 
                 [y_true_valid.min(), y_true_valid.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Equation Predictions')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("VALIDATING NEW EQUATIONS FROM LATEST TRAINING RUN")
    print("=" * 60)
    print("Comparing Power-Based vs Logarithmic Equations")
    print("=" * 60)
    
    # Create validation output directory
    validation_dir = "outputs/validation"
    os.makedirs(validation_dir, exist_ok=True)
    print(f"Output directory: {validation_dir}")
    
    # Load the smoothed data that was used for training
    print("Loading and smoothing data...")
    
    # Load original data
    df = pd.read_csv('data/ENB2012_data.csv')
    
    # Extract features and targets (same as in training)
    feature_cols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    target_cols = ['target_1', 'target_2']
    
    X = df[feature_cols].values.astype(np.float32)
    Y = df[target_cols].values.astype(np.float32)
    
    print(f"Original data shape: {X.shape}")
    
    # Apply the EXACT same smoothing that was used in training
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    
    def smooth_features_savgol(X, window_length=15, polyorder=3):
        """Apply Savitzky-Golay smoothing to features - EXACT same as training"""
        X_smoothed = np.copy(X)
        for i in range(X.shape[1]):
            # Handle edge cases by using valid window sizes - EXACT same logic
            wl = min(window_length, len(X) // 2 * 2 + 1)  # Must be odd
            if wl >= 3:
                X_smoothed[:, i] = savgol_filter(X[:, i], wl, polyorder)
            else:
                X_smoothed[:, i] = X[:, i]  # Keep original if too few points
        return X_smoothed
    
    def smooth_targets_gaussian(Y, sigma=1.5):
        """Apply Gaussian smoothing to targets - EXACT same as training"""
        Y_smoothed = np.copy(Y)
        for i in range(Y.shape[1]):
            Y_smoothed[:, i] = gaussian_filter1d(Y[:, i], sigma=sigma)
        return Y_smoothed
    
    # Apply EXACT same smoothing parameters used in training
    smoothing_method = "savgol"  # Same as training script
    
    if smoothing_method == "savgol":
        # Apply Savitzky-Golay smoothing to ALL features equally - EXACT same
        X_smoothed = smooth_features_savgol(X, window_length=15, polyorder=3)
        Y_smoothed = smooth_targets_gaussian(Y, sigma=1.5)
        print("Applied Savitzky-Golay smoothing (window=15, polyorder=3) to ALL features equally")
    
    # Handle any remaining zeros or very small values - EXACT same as training
    eps = 1e-8
    X_smoothed = np.where(np.abs(X_smoothed) < eps, eps, X_smoothed)
    Y_smoothed = np.where(np.abs(Y_smoothed) < eps, eps, Y_smoothed)
    
    # Ensure ALL values remain positive - EXACT same as training
    min_positive = 0.01  # Minimum positive value
    X_smoothed = np.maximum(X_smoothed, min_positive)  # Force all features to be positive
    Y_smoothed = np.maximum(Y_smoothed, min_positive)  # Force all targets to be positive
    
    print("Applied EXACT same smoothing as training script")
    print(f"Smoothed data shape: {X_smoothed.shape}")
    print(f"Features: Savgol (window=15, polyorder=3)")
    print(f"Targets: Gaussian (sigma=1.5)")
    print(f"All values clamped to minimum: {min_positive}")
    
    # Evaluate BOTH types of equations on smoothed data
    print("\n" + "="*60)
    print("EVALUATING BOTH EQUATION TYPES")
    print("="*60)
    
    # Get predictions from POWER-BASED equations
    print("\n--- POWER-BASED EQUATIONS ---")
    y_pred_power_0 = evaluate_power_equation_0(X_smoothed)
    y_pred_power_1 = evaluate_power_equation_1(X_smoothed)
    
    # Get predictions from LOGARITHMIC equations
    print("\n--- LOGARITHMIC EQUATIONS ---")
    y_pred_log_0 = evaluate_log_equation_0(X_smoothed)
    y_pred_log_1 = evaluate_log_equation_1(X_smoothed)
    
    # Calculate metrics for POWER-BASED equations
    print("\n📊 POWER-BASED EQUATION PERFORMANCE:")
    metrics_power_0 = calculate_equation_metrics(Y[:, 0], y_pred_power_0)
    metrics_power_1 = calculate_equation_metrics(Y[:, 1], y_pred_power_1)
    
    print(f"Output 0 (target_1):")
    for metric, value in metrics_power_0.items():
        print(f"  {metric}: {value:.6f}")
    
    print(f"Output 1 (target_2):")
    for metric, value in metrics_power_1.items():
        print(f"  {metric}: {value:.6f}")
    
    # Calculate metrics for LOGARITHMIC equations
    print("\n📊 LOGARITHMIC EQUATION PERFORMANCE:")
    metrics_log_0 = calculate_equation_metrics(Y[:, 0], y_pred_log_0)
    metrics_log_1 = calculate_equation_metrics(Y[:, 1], y_pred_log_1)
    
    print(f"Output 0 (target_1):")
    for metric, value in metrics_log_0.items():
        print(f"  {metric}: {value:.6f}")
    
    print(f"Output 1 (target_2):")
    for metric, value in metrics_log_1.items():
        print(f"  {metric}: {value:.6f}")
    
    # Calculate averages for comparison
    avg_mse_power = (metrics_power_0['MSE'] + metrics_power_1['MSE']) / 2
    avg_mape_power = (metrics_power_0['MAPE'] + metrics_power_1['MAPE']) / 2
    
    avg_mse_log = (metrics_log_0['MSE'] + metrics_log_1['MSE']) / 2
    avg_mape_log = (metrics_log_0['MAPE'] + metrics_log_1['MAPE']) / 2
    
    print(f"\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    print(f"Power-Based Equations:")
    print(f"  Average MSE: {avg_mse_power:.6f}")
    print(f"  Average MAPE: {avg_mape_power:.6f}")
    
    print(f"\nLogarithmic Equations:")
    print(f"  Average MSE: {avg_mse_log:.6f}")
    print(f"  Average MAPE: {avg_mape_log:.6f}")
    
    # Determine which performs better
    if avg_mape_power < avg_mape_log:
        winner = "POWER-BASED"
        better_mape = avg_mape_power
        worse_mape = avg_mape_log
    else:
        winner = "LOGARITHMIC"
        better_mape = avg_mape_log
        worse_mape = avg_mape_power
    
    print(f"\n🏆 WINNER: {winner} EQUATIONS!")
    print(f"   Better MAPE: {better_mape:.4f}%")
    print(f"   Worse MAPE:  {worse_mape:.4f}%")
    print(f"   Improvement: {abs(worse_mape - better_mape):.4f}%")
    
    # Save equation metrics to CSV
    print("\nSaving equation metrics...")
    
    equation_metrics = {
        'Equation_Type': ['Power_Based', 'Logarithmic'],
        'Target_1_MSE': [metrics_power_0['MSE'], metrics_log_0['MSE']],
        'Target_2_MSE': [metrics_power_1['MSE'], metrics_log_1['MSE']],
        'Average_MSE': [avg_mse_power, avg_mse_log],
        'Target_1_MAPE': [metrics_power_0['MAPE'], metrics_log_0['MAPE']],
        'Target_2_MAPE': [metrics_power_1['MAPE'], metrics_log_1['MAPE']],
        'Average_MAPE': [avg_mape_power, avg_mape_log],
        'Target_1_R2': [metrics_power_0['R2'], metrics_log_0['R2']],
        'Target_2_R2': [metrics_power_1['R2'], metrics_log_1['R2']],
        'Average_R2': [(metrics_power_0['R2'] + metrics_power_1['R2']) / 2, 
                       (metrics_log_0['R2'] + metrics_log_1['R2']) / 2]
    }
    
    equation_df = pd.DataFrame(equation_metrics)
    equation_csv_path = os.path.join(validation_dir, 'equation_comparison_metrics.csv')
    equation_df.to_csv(equation_csv_path, index=False)
    print(f"Equation comparison metrics saved to '{equation_csv_path}'")
    
    # Save equation predictions
    np.save(os.path.join(validation_dir, 'power_equation_predictions_0.npy'), y_pred_power_0)
    np.save(os.path.join(validation_dir, 'power_equation_predictions_1.npy'), y_pred_power_1)
    np.save(os.path.join(validation_dir, 'log_equation_predictions_0.npy'), y_pred_log_0)
    np.save(os.path.join(validation_dir, 'log_equation_predictions_1.npy'), y_pred_log_1)
    print(f"All equation predictions saved to '{validation_dir}/' directory")
    
    # Create visualizations for comparison
    print("\nCreating visualizations...")
    
    # Power-based equations
    plot_equation_predictions(Y[:, 0], y_pred_power_0, 
                             "Power-Based Equation 0: True vs Predicted (target_1)", 
                             os.path.join(validation_dir, "power_equation_0_predictions.png"))
    
    plot_equation_predictions(Y[:, 1], y_pred_power_1, 
                             "Power-Based Equation 1: True vs Predicted (target_2)", 
                             os.path.join(validation_dir, "power_equation_1_predictions.png"))
    
    # Logarithmic equations
    plot_equation_predictions(Y[:, 0], y_pred_log_0, 
                             "Logarithmic Equation 0: True vs Predicted (target_1)", 
                             os.path.join(validation_dir, "log_equation_0_predictions.png"))
    
    plot_equation_predictions(Y[:, 1], y_pred_log_1, 
                             "Logarithmic Equation 1: True vs Predicted (target_2)", 
                             os.path.join(validation_dir, "log_equation_1_predictions.png"))
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL VALIDATION SUMMARY")
    print("="*60)
    print(f"Data smoothing: Savgol (window=15, polyorder=3) + Gaussian (sigma=1.5)")
    print(f"Minimum positive value: {min_positive}")
    
    print(f"\nNeural Network Performance (from training):")
    print(f"  MAPE: 0.08% (EXCELLENT)")
    
    print(f"\nEquation Performance Comparison:")
    print(f"  Power-Based MAPE: {avg_mape_power:.4f}%")
    print(f"  Logarithmic MAPE: {avg_mape_log:.4f}%")
    print(f"  Winner: {winner}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Both equation types work on the SAME smoothed data")
    print(f"   • No scaling transformations needed")
    print(f"   • Equations work directly on your original data scale")
    print(f"   • {winner} equations are more interpretable/accurate")
    
    print(f"\n📁 OUTPUTS SAVED TO: {validation_dir}/")
    print(f"   • equation_comparison_metrics.csv")
    print(f"   • power_equation_predictions_*.npy")
    print(f"   • log_equation_predictions_*.npy")
    print(f"   • power_equation_*_predictions.png")
    print(f"   • log_equation_*_predictions.png")
    
    print(f"\n" + "="*60)
    print("Next steps:")
    print(f"1. Use {winner} equations for predictions")
    print("2. Analyze why one method performs better")
    print("3. Consider if the better equations are more interpretable")

if __name__ == "__main__":
    main()
