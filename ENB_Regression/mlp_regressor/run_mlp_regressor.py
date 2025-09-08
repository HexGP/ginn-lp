#!/usr/bin/env python3
"""
Simple runner script for the MLP regressor model.
"""

import sys
import os

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mlp_regressor import main

if __name__ == "__main__":
    print("Starting MLP Regressor Analysis...")
    print("=" * 50)
    
    try:
        model, results, comparison_results = main()
        print("\n" + "=" * 50)
        print("Analysis completed successfully!")
        print("Check the following files for results:")
        print("- *_results.csv: Detailed metrics for the best model")
        print("- mlp_architecture_comparison.csv: Comparison of all architectures")
        print("- *_regression_results.png: Visualization plots")
        print("- *_loss_curves.png: Training loss curves")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1) 