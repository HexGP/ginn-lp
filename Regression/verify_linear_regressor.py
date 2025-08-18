#!/usr/bin/env python3
"""
Simple verification script to test that the linear regressor 
works with the new standardized data handling approach.
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

def test_linear_regressor():
    """Test that the linear regressor works with standardized data handling."""
    
    print("Testing linear regressor with standardized data handling...")
    
    try:
        # Import the linear regressor
        from linear_regressor.linear_regressor import main
        
        # Run the main function
        print("Running linear regressor...")
        model, results = main()
        
        print("✓ Linear regressor completed successfully!")
        print(f"✓ Model type: {type(model).__name__}")
        print(f"✓ Results keys: {list(results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Linear regressor failed: {e}")
        return False

if __name__ == "__main__":
    success = test_linear_regressor()
    sys.exit(0 if success else 1) 