#!/usr/bin/env python3
"""
Quick script to test the extracted equations from GINN model (4 PTA Blocks)
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

def main():
    print("🔍 Testing GINN Extracted Equations (4 PTA Blocks) on ENB2012 Dataset")
    print("=" * 70)
    
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
    
    # Apply smoothing (same as in training)
    print("\n🔄 Applying Savitzky-Golay smoothing...")
    X_smooth = savgol_positive(X_raw)
    Y_smooth = savgol_positive(Y_raw)
    
    # Create symbols
    symbols = sp.symbols([f"X_{i+1}" for i in range(len(feature_cols))])
    
    # The extracted equations from your 4 PTA block GINN model
    print("\n📝 Testing extracted equations...")
    
    # Output 0 equation (Target 1) - from 4 PTA blocks
    eq_0 = """-0.0252*(X_1**0.161*X_3**1.668*X_4**0.1*X_5**0.848*X_6**1.078*X_8**1.34 + 0.531*X_1**0.238*X_2**0.536*X_3**1.05*X_4**0.59*X_5**0.08*X_6**0.874*X_7**1.44*X_8**1.24 + 0.909*X_1**0.818*X_2**0.516*X_3**0.793*X_4**0.654*X_5**1.04*X_6**2.281*X_7**0.677*X_8**0.905 + 0.593*X_1**1.11*X_2**0.509*X_4**0.511*X_5**0.199*X_6**2.294*X_7**0.513*X_8**0.083)**0.803/(X_1**0.32*X_2**0.248*X_3**0.511*X_4**0.484*X_5**0.224*X_6**1.567*X_7**0.367*X_8**0.793) + 0.398*(X_1**0.161*X_3**1.668*X_4**0.1*X_5**0.848*X_6**1.078*X_8**1.34 + 0.531*X_1**0.238*X_2**0.536*X_3**1.05*X_4**0.59*X_5**0.08*X_6**0.874*X_7**1.44*X_8**1.24 + 0.909*X_1**0.818*X_2**0.516*X_3**0.793*X_4**0.654*X_5**1.04*X_6**2.281*X_7**0.677*X_8**0.905 + 0.593*X_1**1.11*X_2**0.509*X_4**0.511*X_5**0.199*X_6**2.294*X_7**0.513*X_8**0.083)**0.741/(X_1**0.296*X_2**0.229*X_3**0.472*X_4**0.447*X_5**0.207*X_6**1.446*X_7**0.339*X_8**0.732) + 0.0723*(X_1**0.161*X_3**1.668*X_4**0.1*X_5**0.848*X_6**1.078*X_8**1.34 + 0.531*X_1**0.238*X_2**0.536*X_3**1.05*X_4**0.59*X_5**0.08*X_6**0.874*X_7**1.44*X_8**1.24 + 0.909*X_1**0.818*X_2**0.516*X_3**0.793*X_4**0.654*X_5**1.04*X_6**2.281*X_7**0.677*X_8**0.905 + 0.593*X_1**1.11*X_2**0.509*X_4**0.511*X_5**0.199*X_6**2.294*X_7**0.513*X_8**0.083)**0.269/(X_1**0.107*X_2**0.0831*X_3**0.171*X_4**0.162*X_5**0.0751*X_6**0.5251*X_7**0.123*X_8**0.266) - 0.0377*(X_1**0.161*X_3**1.668*X_4**0.1*X_5**0.848*X_6**1.078*X_8**1.34 + 0.531*X_1**0.238*X_2**0.536*X_3**1.05*X_4**0.59*X_5**0.08*X_6**0.874*X_7**1.44*X_8**1.24 + 0.909*X_1**0.818*X_2**0.516*X_3**0.793*X_4**0.654*X_5**1.04*X_6**2.281*X_7**0.677*X_8**0.905 + 0.593*X_1**1.11*X_2**0.509*X_4**0.511*X_5**0.199*X_6**2.294*X_7**0.513*X_8**0.083)**0.257/(X_1**0.103*X_2**0.0795*X_3**0.164*X_4**0.155*X_5**0.0718*X_6**0.5017*X_7**0.117*X_8**0.254) + 0.197"""
    
    # Output 1 equation (Target 2) - from 4 PTA blocks
    eq_1 = """0.274*(X_1**0.161*X_3**1.668*X_4**0.1*X_5**0.848*X_6**1.078*X_8**1.34 + 0.531*X_1**0.238*X_2**0.536*X_3**1.05*X_4**0.59*X_5**0.08*X_6**0.874*X_7**1.44*X_8**1.24 + 0.909*X_1**0.818*X_2**0.516*X_3**0.793*X_4**0.654*X_5**1.04*X_6**2.281*X_7**0.677*X_8**0.905 + 0.593*X_1**1.11*X_2**0.509*X_4**0.511*X_5**0.199*X_6**2.294*X_7**0.513*X_8**0.083)**0.873/(X_1**0.348*X_2**0.27*X_3**0.556*X_4**0.526*X_5**0.244*X_6**1.704*X_7**0.399*X_8**0.863) + 0.1*(X_1**0.161*X_3**1.668*X_4**0.1*X_5**0.848*X_6**1.078*X_8**1.34 + 0.531*X_1**0.238*X_2**0.536*X_3**1.05*X_4**0.59*X_5**0.08*X_6**0.874*X_7**1.44*X_8**1.24 + 0.909*X_1**0.818*X_2**0.516*X_3**0.793*X_4**0.654*X_5**1.04*X_6**2.281*X_7**0.677*X_8**0.905 + 0.593*X_1**1.11*X_2**0.509*X_4**0.511*X_5**0.199*X_6**2.294*X_7**0.513*X_8**0.083)**0.705/(X_1**0.281*X_2**0.218*X_3**0.449*X_4**0.425*X_5**0.197*X_6**1.376*X_7**0.322*X_8**0.697) + 0.105*(X_1**0.161*X_3**1.668*X_4**0.1*X_5**0.848*X_6**1.078*X_8**1.34 + 0.531*X_1**0.238*X_2**0.536*X_3**1.05*X_4**0.59*X_5**0.08*X_6**0.874*X_7**1.44*X_8**1.24 + 0.909*X_1**0.818*X_2**0.516*X_3**0.793*X_4**0.654*X_5**1.04*X_6**2.281*X_7**0.677*X_8**0.905 + 0.593*X_1**1.11*X_2**0.509*X_4**0.511*X_5**0.199*X_6**2.294*X_7**0.513*X_8**0.083)**0.284/(X_1**0.113*X_2**0.0878*X_3**0.181*X_4**0.171*X_5**0.0792*X_6**0.5544*X_7**0.13*X_8**0.281) + 0.42*(X_1**0.161*X_3**1.668*X_4**0.1*X_5**0.848*X_6**1.078*X_8**1.34 + 0.531*X_1**0.238*X_2**0.536*X_3**1.05*X_4**0.59*X_5**0.08*X_6**0.874*X_7**1.44*X_8**1.24 + 0.909*X_1**0.818*X_2**0.516*X_3**0.793*X_4**0.654*X_5**1.04*X_6**2.281*X_7**0.677*X_8**0.905 + 0.593*X_1**1.11*X_2**0.509*X_4**0.511*X_5**0.199*X_6**2.294*X_7**0.513*X_8**0.083)**0.173/(X_1**0.0691*X_2**0.0535*X_3**0.11*X_4**0.104*X_5**0.0482*X_6**0.3377*X_7**0.0791*X_8**0.171) - 0.11"""
    
    # Test both equations
    equations = [eq_0, eq_1]
    equation_names = ["Output 0 (Target 1)", "Output 1 (Target 2)"]
    
    print("\n" + "="*70)
    print("EQUATION EVALUATION RESULTS")
    print("="*70)
    
    for i, (eq, name) in enumerate(zip(equations, equation_names)):
        print(f"\n🔍 Testing {name}:")
        print("-" * 50)
        
        # Evaluate equation
        predictions = evaluate_equation(eq, X_smooth, symbols)
        
        if predictions is not None:
            # Calculate metrics
            mse = mean_squared_error(Y_smooth[:, i], predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(Y_smooth[:, i], predictions)
            mape = calculate_mape(Y_smooth[:, i], predictions)
            r2 = r2_score(Y_smooth[:, i], predictions)
            
            # Display results
            print(f"📊 Metrics:")
            print(f"   MSE:  {mse:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   MAE:  {mae:.4f}")
            print(f"   MAPE: {mape:.2f}%")
            print(f"   R²:   {r2:.4f}")
            
            # Show sample predictions vs actual
            print(f"\n📈 Sample Predictions (first 5):")
            print(f"   Actual:    {Y_smooth[:5, i]}")
            print(f"   Predicted: {predictions[:5]}")
            print(f"   Difference: {Y_smooth[:5, i] - predictions[:5]}")
            
            # Check prediction range
            pred_min, pred_max = np.min(predictions), np.max(predictions)
            actual_min, actual_max = np.min(Y_smooth[:, i]), np.max(Y_smooth[:, i])
            print(f"\n📏 Prediction Range:")
            print(f"   Actual:    [{actual_min:.2f}, {actual_max:.2f}]")
            print(f"   Predicted: [{pred_min:.2f}, {pred_max:.2f}]")
            
        else:
            print(f"❌ Failed to evaluate equation")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✅ Equations evaluated with same smoothing as training")
    print("📊 Metrics calculated: MSE, RMSE, MAE, MAPE, R²")
    print("🔍 Sample predictions and ranges displayed")
    print("="*70)

if __name__ == "__main__":
    main()
