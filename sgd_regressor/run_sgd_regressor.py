#!/usr/bin/env python3
"""
Simple runner script for the multi-task SGD regressor model.
"""

import sys
import os

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sgd_regressor import main

if __name__ == "__main__":
    print("Starting Multi-Task SGD Regressor Analysis...")
    print("=" * 50)
    
    try:
        model, results, comparison_results = main()
        print("\n" + "=" * 50)
        print("Analysis completed successfully!")
        print("Check the following files for results:")
        print("- sgd_regression_results.csv: Detailed metrics")
        print("- sgd_vs_linear_comparison.csv: Comparison with Linear Regression")
        print("- sgd_regression_results.png: Visualization plots")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1) 