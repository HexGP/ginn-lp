#!/usr/bin/env python3
"""
Simple runner script for the multi-output regression model.
"""

import sys
import os

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multiout_regressor import main

if __name__ == "__main__":
    print("Starting Multi-Output Regression Analysis...")
    print("=" * 50)
    
    try:
        model, results, comparison_results = main()
        print("\n" + "=" * 50)
        print("Analysis completed successfully!")
        print("Check the following files for results:")
        print("- *_results.csv: Detailed metrics for the best model")
        print("- model_comparison_results.csv: Comparison of all models")
        print("- *_regression_results.png: Visualization plots")
        print("- feature_importance.png: Feature importance plots (if available)")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1) 