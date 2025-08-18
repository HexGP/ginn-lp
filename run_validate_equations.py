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
    shared_term = (0.6561*X_1**0.667*X_2**0.61*X_3**0.921*X_4**1.079*X_5**1.576*X_6**0.677*X_7**0.945*X_8**0.547 + 
                   0.3896*X_1**1.051*X_2**0.633*X_3**0.119*X_4**1.154*X_5**2.012*X_6**0.46*X_8**1.17 + 
                   X_1**6.713*X_2**0.603*X_3**0.417*X_4**2.523*X_6**1.2*X_7**2.747)
    
    # Apply the discovered equation for output 0 (from latest extraction)
    term1 = 0.264 * (shared_term**0.803) / (X_1**1.379*X_4**1.793*X_5**0.243*X_6**0.912*X_7**0.59*X_8**0.385)
    term2 = 0.2713 * (shared_term**0.741) / (X_1**1.273*X_4**1.655*X_5**0.225*X_6**0.842*X_7**0.545*X_8**0.355)
    term3 = -0.9674 * (shared_term**0.269) / (X_1**0.4621*X_4**0.6006*X_5**0.0815*X_6**0.306*X_7**0.198*X_8**0.129)
    term4 = 1.038 * (shared_term**0.257) / (X_1**0.4416*X_4**0.5739*X_5**0.0779*X_6**0.292*X_7**0.189*X_8**0.123)
    constant = -2.208
    
    y_pred = term1 + term2 + term3 + term4 + constant
    
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
    shared_term = (0.6561*X_1**0.667*X_2**0.61*X_3**0.921*X_4**1.079*X_5**1.576*X_6**0.677*X_7**0.945*X_8**0.547 + 
                   0.3896*X_1**1.051*X_2**0.633*X_3**0.119*X_4**1.154*X_5**2.012*X_6**0.46*X_8**1.17 + 
                   X_1**6.713*X_2**0.603*X_3**0.417*X_4**2.523*X_6**1.2*X_7**2.747)
    
    # Apply the discovered equation for output 1 (from latest extraction)
    term1 = 1.451 * (shared_term**0.873) / (X_1**1.5*X_4**1.949*X_5**0.265*X_6**0.992*X_7**0.642*X_8**0.418)
    term2 = -0.6319 * (shared_term**0.705) / (X_1**1.211*X_4**1.574*X_5**0.214*X_6**0.801*X_7**0.518*X_8**0.338)
    term3 = -0.21 * (shared_term**0.284) / (X_1**0.4879*X_4**0.6342*X_5**0.0861*X_6**0.323*X_7**0.209*X_8**0.136)
    term4 = 0.07226 * (shared_term**0.173) / (X_1**0.2972*X_4**0.3863*X_5**0.0524*X_6**0.196*X_7**0.127*X_8**0.083)
    constant = 1.698
    
    y_pred = term1 + term2 + term3 + term4 + constant
    
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
    print("VALIDATING GINN EQUATIONS FROM LATEST TRAINING RUN")
    print("=" * 60)
    print("Testing Power-Based Equations Extracted from GINN")
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
    
    # EXACT same constants used in training
    MIN_POSITIVE = 1e-2  # Same as training
    EPS_LAURENT = 1e-12  # Same as training
    
    def savgol_positive(X, window_length=15, polyorder=3, min_positive=MIN_POSITIVE):
        """
        Savitzky–Golay smoothing; then clamp to strictly positive floor.
        EXACT same function used in training.
        """
        arr = np.asarray(X, dtype=float).copy()
        n, d = arr.shape
        wl = max(3, min(window_length, (n // 2) * 2 + 1))  # must be odd, <= n and >=3
        for j in range(d):
            if n >= wl:
                arr[:, j] = savgol_filter(arr[:, j], wl, polyorder)
        # clamp: (i) avoid true zeros, (ii) avoid negatives for Laurent stability
        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr = np.sign(arr) * np.maximum(np.abs(arr), EPS_LAURENT)   # avoid exact 0
        arr = np.maximum(arr, min_positive)                         # enforce positive domain
        return arr
    
    # Apply EXACT same smoothing as training
    X_smoothed = savgol_positive(X, window_length=15, polyorder=3, min_positive=MIN_POSITIVE)
    Y_smoothed = savgol_positive(Y, window_length=15, polyorder=3, min_positive=MIN_POSITIVE)
    
    print("Applied EXACT same smoothing as training script")
    print(f"Smoothed data shape: {X_smoothed.shape}")
    print(f"Features: Savgol (window=15, polyorder=3)")
    print(f"Targets: Savgol (window=15, polyorder=3)")
    print(f"MIN_POSITIVE: {MIN_POSITIVE}")
    print(f"EPS_LAURENT: {EPS_LAURENT}")
    
    # Evaluate GINN equations on smoothed data
    print("\n" + "="*60)
    print("EVALUATING GINN EQUATIONS")
    print("="*60)
    
    # Get predictions from GINN equations
    print("\n--- GINN POWER-BASED EQUATIONS ---")
    y_pred_ginn_0 = evaluate_power_equation_0(X_smoothed)
    y_pred_ginn_1 = evaluate_power_equation_1(X_smoothed)
    
    # Calculate metrics for GINN equations
    print("\n📊 GINN EQUATION PERFORMANCE:")
    metrics_ginn_0 = calculate_equation_metrics(Y[:, 0], y_pred_ginn_0)
    metrics_ginn_1 = calculate_equation_metrics(Y[:, 1], y_pred_ginn_1)
    
    print(f"Output 0 (target_1):")
    for metric, value in metrics_ginn_0.items():
        print(f"  {metric}: {value:.6f}")
    
    print(f"Output 1 (target_2):")
    for metric, value in metrics_ginn_1.items():
        print(f"  {metric}: {value:.6f}")
    
    # Calculate averages for summary
    avg_mse_ginn = (metrics_ginn_0['MSE'] + metrics_ginn_1['MSE']) / 2
    avg_mape_ginn = (metrics_ginn_0['MAPE'] + metrics_ginn_1['MAPE']) / 2
    avg_r2_ginn = (metrics_ginn_0['R2'] + metrics_ginn_1['R2']) / 2
    
    print(f"\n" + "="*60)
    print("GINN EQUATION PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Average MSE: {avg_mse_ginn:.6f}")
    print(f"Average MAPE: {avg_mape_ginn:.6f}")
    print(f"Average R²: {avg_r2_ginn:.6f}")
    
    # Save equation metrics to CSV
    print("\nSaving equation metrics...")
    
    equation_metrics = {
        'Equation_Type': ['GINN_Power_Based'],
        'Target_1_MSE': [metrics_ginn_0['MSE']],
        'Target_2_MSE': [metrics_ginn_1['MSE']],
        'Average_MSE': [avg_mse_ginn],
        'Target_1_MAPE': [metrics_ginn_0['MAPE']],
        'Target_2_MAPE': [metrics_ginn_1['MAPE']],
        'Average_MAPE': [avg_mape_ginn],
        'Target_1_R2': [metrics_ginn_0['R2']],
        'Target_2_R2': [metrics_ginn_1['R2']],
        'Average_R2': [avg_r2_ginn]
    }
    
    equation_df = pd.DataFrame(equation_metrics)
    equation_csv_path = os.path.join(validation_dir, 'ginn_equation_metrics.csv')
    equation_df.to_csv(equation_csv_path, index=False)
    print(f"GINN equation metrics saved to '{equation_csv_path}'")
    
    # Save equation predictions
    np.save(os.path.join(validation_dir, 'ginn_equation_predictions_0.npy'), y_pred_ginn_0)
    np.save(os.path.join(validation_dir, 'ginn_equation_predictions_1.npy'), y_pred_ginn_1)
    print(f"GINN equation predictions saved to '{validation_dir}/' directory")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # GINN equations
    plot_equation_predictions(Y[:, 0], y_pred_ginn_0, 
                             "GINN Equation 0: True vs Predicted (target_1)", 
                             os.path.join(validation_dir, "ginn_equation_0_predictions.png"))
    
    plot_equation_predictions(Y[:, 1], y_pred_ginn_1, 
                             "GINN Equation 1: True vs Predicted (target_2)", 
                             os.path.join(validation_dir, "ginn_equation_1_predictions.png"))
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL VALIDATION SUMMARY")
    print("="*60)
    print(f"Data smoothing: Savgol (window=15, polyorder=3)")
    print(f"MIN_POSITIVE: {MIN_POSITIVE}")
    print(f"EPS_LAURENT: {EPS_LAURENT}")
    
    print(f"\nGINN Equation Performance:")
    print(f"  Target 1 MAPE: {metrics_ginn_0['MAPE']:.4f}%")
    print(f"  Target 2 MAPE: {metrics_ginn_1['MAPE']:.4f}%")
    print(f"  Average MAPE: {avg_mape_ginn:.4f}%")
    print(f"  Average R²: {avg_r2_ginn:.4f}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • GINN equations work on the SAME smoothed data used for training")
    print(f"   • No scaling transformations needed")
    print(f"   • Equations work directly on your original data scale")
    print(f"   • These are the actual equations extracted from your trained GINN model")
    
    print(f"\n📁 OUTPUTS SAVED TO: {validation_dir}/")
    print(f"   • ginn_equation_metrics.csv")
    print(f"   • ginn_equation_predictions_*.npy")
    print(f"   • ginn_equation_*_predictions.png")
    
    print(f"\n" + "="*60)
    print("Next steps:")
    print("1. Use these GINN equations for predictions on new data")
    print("2. Compare equation performance with neural network performance")
    print("3. Analyze the mathematical relationships GINN discovered")

if __name__ == "__main__":
    main()
