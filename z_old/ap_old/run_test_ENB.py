#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to evaluate saved equations from ginn_multi_ENB.json on the same test data split.

This script:
1. Loads the saved equations and test data from the JSON file
2. Evaluates the equations on the exact same test data used during training
3. Compares performance with the original training results
4. Provides detailed metrics and analysis

Requirements:
- numpy, pandas, sympy, scikit-learn
- The JSON file from run_cv.py must contain test_data section
"""

import os
import json
import numpy as np
import pandas as pd
import sympy as sp
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = np.abs(y_true) > 1e-10
    if not np.any(mask):
        return np.inf
    
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
    return mape

def metrics(y, yhat):
    """Calculate comprehensive metrics"""
    return dict(
        R2=float(r2_score(y, yhat)),
        MAE=float(mean_absolute_error(y, yhat)),
        MSE=float(mean_squared_error(y, yhat)),
        RMSE=float(np.sqrt(mean_squared_error(y, yhat))),
        MAPE=float(calculate_mape(y, yhat))
    )

def make_vector_fn(exprs, n_features, symbols=None):
    """Create evaluation function for equations"""
    if symbols is None:
        symbols = sp.symbols([f"X_{i+1}" for i in range(n_features)])
    
    # Prepare lambdas once
    exprs = [sp.simplify(e) for e in exprs]
    fns = [sp.lambdify(symbols, e, "numpy") for e in exprs]
    
    def _to_1d_float(y, n):
        """Coerce any result into a 1-D float ndarray of length n"""
        if isinstance(y, sp.Matrix):
            y = np.asarray(y.tolist(), dtype=float)
        else:
            y = np.asarray(y, dtype=float)
        
        if y.ndim == 0:
            y = np.full(n, float(y))
        elif y.ndim == 1:
            if y.shape[0] != n:
                if y.shape[0] == 1:
                    y = np.full(n, float(y[0]))
                else:
                    raise ValueError(f"Expected length {n}, got {y.shape[0]}")
        else:
            y = np.squeeze(y)
            if y.ndim == 0:
                y = np.full(n, float(y))
            elif y.ndim == 1 and y.shape[0] != n:
                if y.shape[0] == 1:
                    y = np.full(n, float(y[0]))
                else:
                    raise ValueError(f"Squeezed shape mismatch: {y.shape}")
            elif y.ndim > 1:
                y = y.reshape(-1)
                if y.shape[0] != n:
                    raise ValueError(f"Flatten shape mismatch: {y.shape}")
        
        # ensure finite
        if not np.isfinite(y).all():
            y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        return y.astype(float, copy=False)
    
    def f(X):
        n = X.shape[0]
        cols = [X[:, i] for i in range(n_features)]
        outs = []
        for fn in fns:
            yi = fn(*cols)
            yi = _to_1d_float(yi, n)
            outs.append(yi)
        return np.column_stack(outs)
    
    return f

def load_json_results(json_path):
    """Load results from JSON file"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    print(f"✅ Loaded results from: {json_path}")
    print(f"📊 Found {len(results)} fold(s)")
    
    return results

def verify_test_data(results):
    """Verify that test data is available in the results"""
    if not results:
        raise ValueError("No results found in JSON file")
    
    first_fold = results[0]
    if 'test_data' not in first_fold:
        raise ValueError("No test_data found in JSON. Please run run_cv.py first to generate this data.")
    
    test_data = first_fold['test_data']
    required_keys = ['X_test', 'Y_test', 'test_indices', 'train_indices']
    missing_keys = [key for key in required_keys if key not in test_data]
    
    if missing_keys:
        raise ValueError(f"Missing required test data keys: {missing_keys}")
    
    print(f"✅ Test data verification passed")
    print(f"   📊 Test set size: {len(test_data['X_test'])} samples")
    print(f"   🎯 Number of targets: {len(test_data['Y_test'][0])}")
    
    return test_data

def evaluate_equations(exprs, X_test, Y_test, target_names):
    """Evaluate equations on test data"""
    n_features = X_test.shape[1]
    n_outputs = Y_test.shape[1]
    
    print(f"\n🔍 Evaluating {len(exprs)} equations on test data...")
    print(f"   📊 Features: {n_features}, Outputs: {n_outputs}")
    
    # Create evaluation function
    try:
        f_vec = make_vector_fn(exprs, n_features)
        Yhat_eq = f_vec(X_test)
        print(f"✅ Equation evaluation successful: y_eq shape = {Yhat_eq.shape}")
    except Exception as e:
        print(f"❌ Equation evaluation failed: {e}")
        return None
    
    # Calculate metrics for each target
    results = []
    for j in range(n_outputs):
        target_name = target_names[j] if j < len(target_names) else f"target_{j+1}"
        
        # Calculate metrics
        m_eq = metrics(Y_test[:, j], Yhat_eq[:, j])
        
        # Store results
        result = {
            'target': target_name,
            'metrics': m_eq,
            'predictions': Yhat_eq[:, j].tolist(),
            'truth': Y_test[:, j].tolist()
        }
        results.append(result)
        
        # Print results
        print(f"\n📊 {target_name}:")
        print(f"   R²: {m_eq['R2']:.4f}")
        print(f"   MAE: {m_eq['MAE']:.4f}")
        print(f"   MSE: {m_eq['MSE']:.4f}")
        print(f"   RMSE: {m_eq['RMSE']:.4f}")
        print(f"   MAPE: {m_eq['MAPE']:.2f}%")
    
    return results

def compare_with_training_results(test_results, training_results):
    """Compare test results with original training results"""
    print("\n" + "="*70)
    print("COMPARISON WITH TRAINING RESULTS")
    print("="*70)
    
    if not training_results:
        print("⚠️  No training results to compare with")
        return
    
    first_fold = training_results[0]
    if 'per_target' not in first_fold:
        print("⚠️  No per_target results found in training data")
        return
    
    training_per_target = first_fold['per_target']
    
    for i, test_result in enumerate(test_results):
        if i >= len(training_per_target):
            break
            
        target_name = test_result['target']
        training_target = training_per_target[i]
        
        print(f"\n🔍 {target_name}:")
        print(f"   📊 Test Results (Current Run):")
        print(f"      R²: {test_result['metrics']['R2']:.4f} | MAE: {test_result['metrics']['MAE']:.4f} | MSE: {test_result['metrics']['MSE']:.4f}")
        
        if 'eq_refit' in training_target:
            print(f"   📊 Training Results (Original):")
            print(f"      R²: {training_target['eq_refit']['R2']:.4f} | MAE: {training_target['eq_refit']['MAE']:.4f} | MSE: {training_target['eq_refit']['MSE']:.4f}")
            
            # Check if results are consistent
            r2_diff = abs(test_result['metrics']['R2'] - training_target['eq_refit']['R2'])
            if r2_diff < 1e-6:
                print(f"   ✅ Results are IDENTICAL (R² difference: {r2_diff:.2e})")
            else:
                print(f"   ⚠️  Results differ (R² difference: {r2_diff:.6f})")
        else:
            print(f"   ⚠️  No refit equation results found in training data")

def save_test_results(test_results, output_path):
    """Save test results to a new JSON file"""
    output_data = {
        'test_evaluation': test_results,
        'timestamp': pd.Timestamp.now().isoformat(),
        'script': 'run_test_ENB.py'
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Test results saved to: {output_path}")

def main():
    """Main function"""
    print("🚀 GINN Equation Test Script")
    print("="*50)
    
    # Configuration
    json_path = "outputs/ginn_multi_ENB.json"
    output_path = "outputs/test_results_ENB.json"
    
    try:
        # 1. Load results from JSON
        results = load_json_results(json_path)
        
        # 2. Verify test data is available
        test_data = verify_test_data(results)
        
        # 3. Extract test data
        X_test = np.array(test_data['X_test'])
        Y_test = np.array(test_data['Y_test'])
        
        print(f"\n📊 Test data loaded:")
        print(f"   Features: {X_test.shape}")
        print(f"   Targets: {Y_test.shape}")
        
        # 4. Get equations from the first fold
        first_fold = results[0]
        if 'per_target' not in first_fold:
            raise ValueError("No per_target results found in JSON")
        
        # Extract refit equations (the important ones)
        exprs = []
        target_names = []
        for target_data in first_fold['per_target']:
            if 'expr_refit' in target_data:
                exprs.append(target_data['expr_refit'])
                target_names.append(target_data['target'])
            else:
                print(f"⚠️  No refit equation found for {target_data.get('target', 'unknown')}")
        
        if not exprs:
            raise ValueError("No refit equations found in JSON")
        
        print(f"\n📝 Loaded {len(exprs)} refit equations:")
        for i, (expr, target) in enumerate(zip(exprs, target_names)):
            print(f"   {i+1}. {target}: {str(expr)[:100]}...")
        
        # 5. Evaluate equations on test data
        test_results = evaluate_equations(exprs, X_test, Y_test, target_names)
        
        if test_results is None:
            print("❌ Equation evaluation failed, cannot proceed")
            return
        
        # 6. Compare with training results
        compare_with_training_results(test_results, results)
        
        # 7. Save test results
        save_test_results(test_results, output_path)
        
        # 8. Summary
        print("\n" + "="*70)
        print("🎯 TEST EVALUATION COMPLETE")
        print("="*70)
        
        avg_r2 = np.mean([r['metrics']['R2'] for r in test_results])
        avg_mape = np.mean([r['metrics']['MAPE'] for r in test_results])
        
        print(f"📊 Average Performance:")
        print(f"   R²: {avg_r2:.4f}")
        print(f"   MAPE: {avg_mape:.2f}%")
        
        if avg_r2 >= 0.8:
            print(f"✅ EXCELLENT performance! Equations generalize well to test data.")
        elif avg_r2 >= 0.6:
            print(f"🟡 GOOD performance! Equations show reasonable generalization.")
        elif avg_r2 >= 0.4:
            print(f"🟠 MODERATE performance! Equations need improvement.")
        else:
            print(f"🔴 POOR performance! Equations don't generalize well.")
        
        print("\n📚 Next steps:")
        print("   1. Analyze which equations perform best/worst")
        print("   2. Consider equation complexity vs performance trade-off")
        print("   3. Run additional validation if needed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
