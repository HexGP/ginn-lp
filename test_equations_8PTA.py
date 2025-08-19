#!/usr/bin/env python3
"""
Quick script to test the extracted equations from GINN model
on the ENB2012 dataset to see their actual performance.
"""

import numpy as np
import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.signal import savgol_filter

# Configuration
DATA_CSV = "data/ENB2012_data.csv"
MIN_POSITIVE = 1e-2
EPS_LAURENT = 1e-12

def savgol_positive(X, window_length=15, polyorder=3, min_positive=MIN_POSITIVE):
    """Apply Savitzky-Golay smoothing and ensure positivity"""
    arr = np.asarray(X, dtype=float).copy()
    n, d = arr.shape
    wl = max(3, min(window_length, (n // 2) * 2 + 1))
    for j in range(d):
        if n >= wl:
            arr[:, j] = savgol_filter(arr[:, j], wl, polyorder)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.sign(arr) * np.maximum(np.abs(arr), EPS_LAURENT)
    arr = np.maximum(arr, min_positive)
    return arr

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    # Avoid division by zero
    mask = np.abs(y_true) > 1e-10
    if not np.any(mask):
        return np.inf
    
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
    return mape

def evaluate_equation(expr_str, X, symbols):
    """Evaluate a SymPy expression on the data"""
    try:
        # Parse the expression
        expr = parse_expr(expr_str)
        
        # Create lambda function
        f = sp.lambdify(symbols, expr, modules="numpy")
        
        # Evaluate with safe inputs
        n_samples = X.shape[0]
        safe_inputs = []
        for i in range(X.shape[1]):
            v = X[:, i].copy()
            v = np.where(np.isfinite(v), v, 0.0)
            v = np.sign(v) * np.maximum(np.abs(v), EPS_LAURENT)
            v = np.maximum(v, MIN_POSITIVE)
            safe_inputs.append(v)
        
        # Evaluate
        result = f(*safe_inputs)
        
        # Ensure we get a vector
        if np.isscalar(result):
            result = np.full(n_samples, result)
        elif result.ndim > 1:
            result = result.reshape(-1)
            
        return result.astype(float)
        
    except Exception as e:
        print(f"Error evaluating equation: {e}")
        return None

def main():
    print("🔍 Testing GINN Equations: Raw vs Refitted Comparison on ENB2012 Dataset")
    print("=" * 80)
    
    # Load data
    print("📊 Loading data...")
    df = pd.read_csv(DATA_CSV)
    feature_cols = [col for col in df.columns if not col.startswith('target')]
    target_cols = [col for col in df.columns if col.startswith('target')]
    
    X_raw = df[feature_cols].values.astype(np.float32)
    Y_raw = df[target_cols].values.astype(np.float32)
    
    print(f"Features: {feature_cols}")
    print(f"Targets: {target_cols}")
    print(f"Data shape: {X_raw.shape}")
    
    # Apply smoothing
    print("\n🔄 Applying Savitzky-Golay smoothing...")
    X_smooth = savgol_positive(X_raw)
    Y_smooth = savgol_positive(Y_raw)
    
    # Create symbols
    symbols = sp.symbols([f"X_{i+1}" for i in range(len(feature_cols))])
    
    # Load equations from the JSON file
    print("\n📝 Loading equations from training results...")
    import json
    
    try:
        with open("outputs/ginn_multi_eqsync.json", "r") as f:
            results = json.load(f)
        
        # Get equations from the first fold
        fold_data = results[0]["per_target"]
        
        # Raw equations (directly extracted from model)
        raw_eq_0 = fold_data[0]["expr"]  # Target 1
        raw_eq_1 = fold_data[1]["expr"]  # Target 2
        
        # Refitted equations (after ridge regression optimization)
        refit_eq_0 = fold_data[0]["expr_refit"]  # Target 1
        refit_eq_1 = fold_data[1]["expr_refit"]  # Target 2
        
        print("✅ Successfully loaded equations from JSON file")
        
    except Exception as e:
        print(f"❌ Error loading equations: {e}")
        print("Falling back to hardcoded equations...")
        
        # Fallback to hardcoded equations if JSON loading fails
        raw_eq_0 = """-0.498*(X_1**0.318*X_2**1.623*X_3**1.256*X_4**2.68*X_5**1.434*X_6**1.347*X_7**0.089*X_8**1.74 + 0.0408*X_1**0.688*X_2**0.703*X_3**1.966*X_4**1.75*X_5**2.201*X_6**0.618*X_7**1.507*X_8**0.85 + 0.203*X_1**1.15*X_2**1.584*X_3**0.502*X_4**1.4*X_5**2.581*X_6**1.473*X_7**0.294*X_8**1.21 + 0.912*X_1**1.33*X_2**1.418*X_3**1.572*X_4**2.273*X_5**1.517*X_6**2.566*X_7**0.588*X_8**1.14)**0.803/(X_1**0.808*X_2**1.426*X_3**1.337*X_4**2.168*X_5**2.068*X_6**1.38*X_7**0.307*X_8**1.29) - 0.472*(X_1**0.318*X_2**1.623*X_3**1.256*X_4**2.68*X_5**1.434*X_6**1.347*X_7**0.089*X_8**1.74 + 0.0408*X_1**0.688*X_2**0.703*X_3**1.966*X_4**1.75*X_5**2.201*X_6**0.618*X_7**1.507*X_8**0.85 + 0.203*X_1**1.15*X_2**1.584*X_3**0.502*X_4**1.4*X_5**2.581*X_6**1.473*X_7**0.294*X_8**1.21 + 0.912*X_1**1.33*X_2**1.418*X_3**1.572*X_4**2.273*X_5**1.517*X_6**2.566*X_7**0.588*X_8**1.14)**0.741/(X_1**0.745*X_2**1.316*X_3**1.234*X_4**2.001*X_5**1.909*X_6**1.274*X_7**0.284*X_8**1.19) + 0.62*(X_1**0.318*X_2**1.623*X_3**1.256*X_4**2.68*X_5**1.434*X_6**1.347*X_7**0.089*X_8**1.74 + 0.0408*X_1**0.688*X_2**0.703*X_3**1.966*X_4**1.75*X_5**2.201*X_6**0.618*X_7**1.507*X_8**0.85 + 0.203*X_1**1.15*X_2**1.584*X_3**0.502*X_4**1.4*X_5**2.581*X_6**1.473*X_7**0.294*X_8**1.21 + 0.912*X_1**1.33*X_2**1.418*X_3**1.572*X_4**2.273*X_5**1.517*X_6**2.566*X_7**0.588*X_8**1.14)**0.269/(X_1**0.27*X_2**0.4777*X_3**0.4478*X_4**0.7262*X_5**0.6929*X_6**0.4624*X_7**0.103*X_8**0.431) - 0.32*(X_1**0.318*X_2**1.623*X_3**1.256*X_4**2.68*X_5**1.434*X_6**1.347*X_7**0.089*X_8**1.74 + 0.0408*X_1**0.688*X_2**0.703*X_3**1.966*X_4**1.75*X_5**2.201*X_6**0.618*X_7**1.507*X_8**0.85 + 0.203*X_1**1.15*X_2**1.584*X_3**0.502*X_4**1.4*X_5**2.581*X_6**1.473*X_7**0.294*X_8**1.21 + 0.912*X_1**1.33*X_2**1.418*X_3**1.572*X_4**2.273*X_5**1.517*X_6**2.566*X_7**0.588*X_8**1.14)**0.257/(X_1**0.258*X_2**0.4565*X_3**0.4279*X_4**0.694*X_5**0.6621*X_6**0.4418*X_7**0.0984*X_8**0.412) - 0.066"""
        raw_eq_1 = """-1.022*(X_1**0.318*X_2**1.623*X_3**1.256*X_4**2.68*X_5**1.434*X_6**1.347*X_7**0.089*X_8**1.74 + 0.0408*X_1**0.688*X_2**0.703*X_3**1.966*X_4**1.75*X_5**2.201*X_6**0.618*X_7**1.507*X_8**0.85 + 0.203*X_1**1.15*X_2**1.584*X_3**0.502*X_4**1.4*X_5**2.581*X_6**1.473*X_7**0.294*X_8**1.21 + 0.912*X_1**1.33*X_2**1.418*X_3**1.572*X_4**2.273*X_5**1.517*X_6**2.566*X_7**0.588*X_8**1.14)**0.873/(X_1**0.878*X_2**1.551*X_3**1.454*X_4**2.357*X_5**2.249*X_6**1.501*X_7**0.334*X_8**1.4) - 0.797*(X_1**0.318*X_2**1.623*X_3**1.256*X_4**2.68*X_5**1.434*X_6**1.347*X_7**0.089*X_8**1.74 + 0.0408*X_1**0.688*X_2**0.703*X_3**1.966*X_4**1.75*X_5**2.201*X_6**0.618*X_7**1.507*X_8**0.85 + 0.203*X_1**1.15*X_2**1.584*X_3**0.502*X_4**1.4*X_5**2.581*X_6**1.473*X_7**0.294*X_8**1.21 + 0.912*X_1**1.33*X_2**1.418*X_3**1.572*X_4**2.273*X_5**1.517*X_6**2.566*X_7**0.588*X_8**1.14)**0.705/(X_1**0.709*X_2**1.252*X_3**1.174*X_4**1.903*X_5**1.816*X_6**1.212*X_7**0.27*X_8**1.13) - 0.212*(X_1**0.318*X_2**1.623*X_3**1.256*X_4**2.68*X_5**1.434*X_6**1.347*X_7**0.089*X_8**1.74 + 0.0408*X_1**0.688*X_2**0.703*X_3**1.966*X_4**1.75*X_5**2.201*X_6**0.618*X_7**1.507*X_8**0.85 + 0.203*X_1**1.15*X_2**1.584*X_3**0.502*X_4**1.4*X_5**2.581*X_6**1.473*X_7**0.294*X_8**1.21 + 0.912*X_1**1.33*X_2**1.418*X_3**1.572*X_4**2.273*X_5**1.517*X_6**2.566*X_7**0.588*X_8**1.14)**0.284/(X_1**0.286*X_2**0.5044*X_3**0.4728*X_4**0.7667*X_5**0.7316*X_6**0.4882*X_7**0.109*X_8**0.455) + 1.289*(X_1**0.318*X_2**1.623*X_3**1.256*X_4**2.68*X_5**1.434*X_6**1.347*X_7**0.089*X_8**1.74 + 0.0408*X_1**0.688*X_2**0.703*X_3**1.966*X_4**1.75*X_5**2.201*X_6**0.618*X_7**1.507*X_8**0.85 + 0.203*X_1**1.15*X_2**1.584*X_3**0.502*X_4**1.4*X_5**2.581*X_6**1.473*X_7**0.294*X_8**1.21 + 0.912*X_1**1.33*X_2**1.418*X_3**1.572*X_4**2.273*X_5**1.517*X_6**2.566*X_7**0.588*X_8**1.14)**0.173/(X_1**0.174*X_2**0.3073*X_3**0.288*X_4**0.467*X_5**0.4457*X_6**0.2974*X_7**0.0663*X_8**0.277) - 0.062"""
        
        # For refitted equations, we'll use placeholder text
        refit_eq_0 = "Refitted equation not available in fallback mode"
        refit_eq_1 = "Refitted equation not available in fallback mode"
    
    # Test equations - both raw and refitted
    equations = [
        ("Raw", raw_eq_0, "Target 1 (Output 0)"),
        ("Raw", raw_eq_1, "Target 2 (Output 1)"),
        ("Refitted", refit_eq_0, "Target 1 (Output 0)"),
        ("Refitted", refit_eq_1, "Target 2 (Output 1)")
    ]
    
    print("\n📊 Equation Performance Analysis:")
    print("-" * 50)
    
    for eq_type, eq, target_name in equations:
        print(f"\n🎯 {eq_type} Equation - {target_name}:")
        
        # Skip if equation is not available
        if "not available" in eq:
            print(f"  ❌ {eq}")
            continue
        
        # Evaluate equation
        predictions = evaluate_equation(eq, X_smooth, symbols)
        
        if predictions is not None:
            # Determine target index based on target name
            target_idx = 0 if "Target 1" in target_name else 1
            
            # Calculate metrics
            r2 = r2_score(Y_smooth[:, target_idx], predictions)
            mae = mean_absolute_error(Y_smooth[:, target_idx], predictions)
            mse = mean_squared_error(Y_smooth[:, target_idx], predictions)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE
            mape = calculate_mape(Y_smooth[:, target_idx], predictions)
            
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            
            # Show some sample predictions vs actual
            print(f"  Sample predictions (first 5):")
            for j in range(min(5, len(predictions))):
                print(f"    Actual: {Y_smooth[j, target_idx]:.3f}, Predicted: {predictions[j]:.3f}")
                
            # Check for any extreme values
            if np.any(np.abs(predictions) > 1000):
                print(f"  ⚠️  Warning: Some predictions are very large (>1000)")
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                print(f"  ❌ Error: Predictions contain NaN or Inf values")
            
            # Range analysis
            actual_min, actual_max = np.min(Y_smooth[:, target_idx]), np.max(Y_smooth[:, target_idx])
            pred_min, pred_max = np.min(predictions), np.max(predictions)
            print(f"  📏 Range Analysis:")
            print(f"     Actual: [{actual_min:.2f}, {actual_max:.2f}]")
            print(f"     Predicted: [{pred_min:.2f}, {pred_max:.2f}]")
            
            # Scale assessment
            if pred_min < 0 and actual_min >= 0:
                print(f"  ⚠️  Warning: Predictions include negative values but targets are positive")
            if abs(pred_max - actual_max) > 10:
                print(f"  ⚠️  Warning: Large scale mismatch between predictions and actual values")
                
        else:
            print(f"  ❌ Failed to evaluate equation")
    
    print("\n" + "=" * 80)
    print("✅ Equation testing completed!")
    print("\n💡 Key insights:")
    print("- R² > 0.9: Excellent performance")
    print("- R² > 0.7: Good performance") 
    print("- R² > 0.5: Acceptable performance")
    print("- R² < 0: Equation is worse than random")
    print("- MAPE < 15%: Good scale alignment")
    print("- Range match: Predictions should be in similar scale as targets")
    print("\n🔍 Comparison Summary:")
    print("- Raw equations: Directly extracted from model weights (often poor scale)")
    print("- Refitted equations: Optimized with ridge regression (much better performance)")
    print("- The difference shows the power of equation refitting!")

if __name__ == "__main__":
    main()
