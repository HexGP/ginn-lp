#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic regression data using real mathematical formulas.
This creates a CSV file that can be used to test the GINN surrogate extraction system.

Features:
- Configurable number of features and targets
- Real mathematical relationships (polynomial, trigonometric, exponential)
- Controlled noise levels
- Reproducible results
"""

import numpy as np
import pandas as pd
import math
from typing import List, Tuple

def generate_feature_data(n_samples: int, n_features: int, noise_level: float = 0.1, 
                         random_state: int = 42) -> np.ndarray:
    """
    Generate feature data with controlled distributions and noise.
    
    Args:
        n_samples: Number of data points
        n_features: Number of features
        noise_level: Standard deviation of noise to add
        random_state: Random seed for reproducibility
    
    Returns:
        Feature matrix of shape (n_samples, n_features)
    """
    np.random.seed(random_state)
    
    # Generate base features with different distributions
    features = []
    
    for i in range(n_features):
        if i % 4 == 0:
            # Normal distribution
            feature = np.random.normal(0, 1, n_samples)
        elif i % 4 == 1:
            # Uniform distribution
            feature = np.random.uniform(-2, 2, n_samples)
        elif i % 4 == 2:
            # Exponential distribution (shifted)
            feature = np.random.exponential(1, n_samples) - 1
        else:
            # Chi-squared distribution (shifted)
            feature = np.random.chisquare(3, n_samples) - 3
        
        # Add noise
        noise = np.random.normal(0, noise_level, n_samples)
        feature = feature + noise
        
        # Clip to reasonable range
        feature = np.clip(feature, -5, 5)
        
        features.append(feature)
    
    return np.column_stack(features)

def create_target_formulas(n_features: int) -> List[str]:
    """
    Create mathematical formulas for targets based on number of features.
    
    Args:
        n_features: Number of input features
    
    Returns:
        List of formula strings
    """
    formulas = []
    
    # Target 1: Complex polynomial with interactions
    if n_features >= 8:
        formula1 = f"""
        (2.5 * X_1**2 + 1.8 * X_2**2 - 0.9 * X_3**2 + 
         1.2 * X_1 * X_2 - 0.7 * X_2 * X_3 + 0.5 * X_3 * X_4 +
         0.3 * X_5 + 0.2 * X_6 - 0.1 * X_7 + 0.05 * X_8 + 1.5)
        """
    elif n_features >= 6:
        formula1 = f"""
        (2.0 * X_1**2 + 1.5 * X_2**2 - 0.8 * X_3**2 + 
         1.0 * X_1 * X_2 - 0.6 * X_2 * X_3 + 0.4 * X_4 + 0.3 * X_5 + 1.2)
        """
    else:
        formula1 = f"""
        (1.5 * X_1**2 + 1.2 * X_2**2 + 0.8 * X_1 * X_2 + 0.5 * X_3 + 1.0)
        """
    
    # Target 2: Trigonometric and exponential relationships
    if n_features >= 8:
        formula2 = f"""
        (3.0 * np.sin(X_1) + 2.0 * np.cos(X_2) + 1.5 * np.exp(-abs(X_3)) +
         1.0 * X_4 * X_5 + 0.8 * np.tanh(X_6) + 0.6 * X_7 + 0.4 * X_8 + 2.0)
        """
    elif n_features >= 6:
        formula2 = f"""
        (2.5 * np.sin(X_1) + 1.8 * np.cos(X_2) + 1.2 * np.exp(-abs(X_3)) +
         0.8 * X_4 + 0.6 * np.tanh(X_5) + 1.5)
        """
    else:
        formula2 = f"""
        (2.0 * np.sin(X_1) + 1.5 * np.cos(X_2) + 0.8 * np.exp(-abs(X_3)) + 1.0)
        """
    
    formulas = [formula1, formula2]
    return formulas

def evaluate_formula(formula: str, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """
    Evaluate a mathematical formula given feature data.
    
    Args:
        formula: Mathematical formula string
        X: Feature matrix
        feature_names: List of feature names
    
    Returns:
        Array of target values
    """
    # Create local variables for each feature
    local_vars = {}
    for i, name in enumerate(feature_names):
        local_vars[name] = X[:, i]
    
    # Add numpy functions
    local_vars.update({
        'np': np,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'tanh': np.tanh,
        'exp': np.exp,
        'abs': np.abs,
        'sqrt': np.sqrt,
        'log': np.log,
        'pi': np.pi
    })
    
    try:
        # Evaluate the formula
        result = eval(formula, {"__builtins__": {}}, local_vars)
        return np.array(result)
    except Exception as e:
        print(f"Error evaluating formula: {e}")
        print(f"Formula: {formula}")
        return np.zeros(X.shape[0])

def generate_regression_data(n_samples: int = 1000, n_features: int = 8, n_targets: int = 2,
                           noise_level: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate complete regression dataset with features and targets.
    
    Args:
        n_samples: Number of data points
        n_features: Number of features
        n_targets: Number of targets
        noise_level: Noise level for features
        random_state: Random seed
    
    Returns:
        Tuple of (features, targets, target_formulas)
    """
    print(f"Generating {n_samples} samples with {n_features} features and {n_targets} targets...")
    
    # Generate features
    X = generate_feature_data(n_samples, n_features, noise_level, random_state)
    
    # Create feature names
    feature_names = [f'X_{i+1}' for i in range(n_features)]
    
    # Generate target formulas
    target_formulas = create_target_formulas(n_features)
    
    # Limit to requested number of targets
    target_formulas = target_formulas[:n_targets]
    
    # Generate targets
    Y = []
    for i, formula in enumerate(target_formulas):
        print(f"Generating target {i+1} using formula:")
        print(f"  {formula.strip()}")
        
        target_values = evaluate_formula(formula, X, feature_names)
        
        # Add some noise to targets to make it more realistic
        target_noise = np.random.normal(0, noise_level * 0.5, n_samples)
        target_values = target_values + target_noise
        
        Y.append(target_values)
    
    Y = np.column_stack(Y)
    
    print(f"Generated data shapes: X={X.shape}, Y={Y.shape}")
    print(f"Feature ranges: {[(X[:, i].min(), X[:, i].max()) for i in range(min(3, n_features))]}...")
    print(f"Target ranges: {[(Y[:, i].min(), Y[:, i].max()) for i in range(n_targets)]}")
    
    return X, Y, target_formulas

def save_to_csv(X: np.ndarray, Y: np.ndarray, output_path: str = "synthetic_regression_data.csv"):
    """
    Save the generated data to a CSV file.
    
    Args:
        X: Feature matrix
        Y: Target matrix
        output_path: Output file path
    """
    # Create feature names
    feature_names = [f'X_{i+1}' for i in range(X.shape[1])]
    target_names = [f'target_{i+1}' for i in range(Y.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add targets
    for i, target_name in enumerate(target_names):
        df[target_name] = Y[:, i]
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")
    
    # Display sample
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Display statistics
    print("\nData statistics:")
    print(df.describe())

def main():
    """Main function to generate and save regression data."""
    print("🚀 Synthetic Regression Data Generator")
    print("=" * 50)
    
    # Configuration
    N_SAMPLES = 1000      # Number of data points
    N_FEATURES = 8        # Number of features
    N_TARGETS = 2         # Number of targets
    NOISE_LEVEL = 0.1     # Noise level (0.0 = no noise, 0.5 = high noise)
    RANDOM_STATE = 42     # Random seed for reproducibility
    OUTPUT_PATH = "synthetic_regression_data.csv"
    
    print(f"Configuration:")
    print(f"  Samples: {N_SAMPLES}")
    print(f"  Features: {N_FEATURES}")
    print(f"  Targets: {N_TARGETS}")
    print(f"  Noise Level: {NOISE_LEVEL}")
    print(f"  Random State: {RANDOM_STATE}")
    print(f"  Output: {OUTPUT_PATH}")
    
    try:
        # Generate data
        X, Y, formulas = generate_regression_data(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_targets=N_TARGETS,
            noise_level=NOISE_LEVEL,
            random_state=RANDOM_STATE
        )
        
        # Save to CSV
        save_to_csv(X, Y, OUTPUT_PATH)
        
        # Save formulas to a text file for reference
        formula_path = "target_formulas.txt"
        with open(formula_path, 'w') as f:
            f.write("Target Formulas Used to Generate Data:\n")
            f.write("=" * 50 + "\n\n")
            for i, formula in enumerate(formulas):
                f.write(f"Target {i+1}:\n")
                f.write(f"{formula.strip()}\n\n")
        
        print(f"\n✅ Data generation complete!")
        print(f"📊 CSV file: {OUTPUT_PATH}")
        print(f"📝 Formulas saved to: {formula_path}")
        print(f"\n🎯 You can now use this CSV with run_cv.py!")
        
        # Show how to use with run_cv.py
        print(f"\n📚 To use with run_cv.py:")
        print(f"1. Copy {OUTPUT_PATH} to your data/ folder")
        print(f"2. Update DATA_CSV in run_cv.py to point to this file")
        print(f"3. Run: python run_cv.py")
        
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
