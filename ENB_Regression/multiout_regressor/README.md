# Multi-Output Regression for Multi-Task Learning on ENB2012 Dataset

This directory contains a comprehensive Multi-Output Regression framework for multi-task learning on the ENB2012 dataset.

## Overview

The ENB2012 dataset contains building energy efficiency data with:
- **8 input features** (X1-X8): Building characteristics
- **2 target variables** (target_1, target_2): Heating and cooling loads

## Model Features

### Multi-Output Regression Framework (`multiout_regressor.py`)

A comprehensive multi-output regression framework that:
- Handles multiple target variables simultaneously using scikit-learn's MultiOutputRegressor
- Supports multiple base estimators (Random Forest, Gradient Boosting, Linear, Ridge, Lasso, SVR)
- Automatically compares different models and selects the best performer
- Uses feature scaling for better performance
- Provides detailed evaluation metrics (MSE, MAPE, RMSE)
- Includes cross-validation
- Generates visualizations
- Extracts feature importance (for tree-based models)
- Extracts coefficients (for linear models)

## Supported Base Estimators

The framework supports the following base estimators:

1. **Random Forest**: Ensemble of decision trees with feature importance
2. **Gradient Boosting**: Sequential ensemble with adaptive learning
3. **Linear Regression**: Standard linear regression
4. **Ridge Regression**: Linear regression with L2 regularization
5. **Lasso Regression**: Linear regression with L1 regularization
6. **Support Vector Regression (SVR)**: Non-linear regression with kernel methods

## Key Advantages

1. **Model Comparison**: Automatically tests multiple algorithms
2. **Feature Importance**: Available for tree-based models
3. **Coefficient Analysis**: Available for linear models
4. **Scalability**: Efficient for large datasets
5. **Flexibility**: Easy to add new base estimators

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Multi-Output Regression Model

```bash
python run_multiout_regressor.py
```

Or directly:

```bash
python multiout_regressor.py
```

### Using the Model Programmatically

```python
from multiout_regressor import MultiOutputRegressionModel, load_and_preprocess_data

# Load data
X, y, feature_names, target_names = load_and_preprocess_data('../data/ENB2012_data.csv')

# Create and fit model with specific estimator
model = MultiOutputRegressionModel(
    base_estimator='random_forest',  # or 'gradient_boosting', 'linear', 'ridge', 'lasso', 'svr'
    random_state=42
)
model.fit(X, y, feature_names, target_names)

# Make predictions
predictions = model.predict(X)

# Get feature importance (for tree-based models)
try:
    importance = model.get_feature_importance()
    print("Feature importance:", importance)
except ValueError:
    print("Feature importance not available for this estimator")

# Get coefficients (for linear models)
try:
    coefficients = model.get_coefficients()
    print("Coefficients:", coefficients)
except ValueError:
    print("Coefficients not available for this estimator")
```

## Model Comparison

The framework automatically compares the following models:

1. **Random Forest**: Good for non-linear relationships, provides feature importance
2. **Gradient Boosting**: Often achieves best performance, handles complex patterns
3. **Linear Regression**: Simple, interpretable, fast
4. **Ridge Regression**: Linear with L2 regularization, prevents overfitting
5. **Lasso Regression**: Linear with L1 regularization, performs feature selection

## Output Files

After running the model, you'll get:

1. **`*_results.csv`**: Detailed performance metrics for the best model
2. **`model_comparison_results.csv`**: Comparison of all tested models
3. **`*_regression_results.png`**: Visualization plots showing true vs predicted values
4. **`feature_importance.png`**: Feature importance plots (for tree-based models)

## Performance Metrics

The models evaluate performance using:

- **MSE (Mean Squared Error)**: Average squared difference between predictions and actual values
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **RMSE (Root Mean Squared Error)**: Square root of MSE

## Model Architecture

### Multi-Output Regression Framework

The framework implements a systematic approach:
1. **Data Preprocessing**: Feature scaling using StandardScaler
2. **Model Comparison**: Tests multiple base estimators
3. **Best Model Selection**: Chooses the model with lowest MSE
4. **Detailed Analysis**: Performs comprehensive evaluation on the best model
5. **Feature Analysis**: Extracts feature importance or coefficients
6. **Cross-Validation**: Validates performance robustness

## Data Preprocessing

- **Feature Scaling**: StandardScaler is used to normalize features
- **Train-Test Split**: 80-20 split with random state for reproducibility
- **Cross-Validation**: 5-fold cross-validation for robust evaluation

## Model Selection Process

The framework follows this process:
1. Tests all supported base estimators
2. Evaluates each model using MSE, MAPE, and RMSE
3. Selects the best performing model by MSE
4. Performs detailed analysis on the selected model
5. Generates comprehensive visualizations and reports

## Example Output

```
MODEL COMPARISON
============================================================

Testing Random Forest...
  Average MSE:  8.2345
  Average MAPE: 0.0934
  Average RMSE: 2.8696

Testing Gradient Boosting...
  Average MSE:  7.5678
  Average MAPE: 0.0856
  Average RMSE: 2.7509

Testing Linear Regression...
  Average MSE:  15.1234
  Average MAPE: 0.1456
  Average RMSE: 3.8895

Best model by MSE: Gradient Boosting (MSE: 7.5678)

MULTI-OUTPUT GRADIENT BOOSTING EVALUATION RESULTS
============================================================

TARGET_1:
------------------------------
     MSE: 7.2345
    MAPE: 0.0834
    RMSE: 2.6896

TARGET_2:
------------------------------
     MSE: 7.9012
    MAPE: 0.0878
    RMSE: 2.8109

AVERAGE ACROSS ALL TARGETS
============================================================
     MSE: 7.5678
    MAPE: 0.0856
    RMSE: 2.7502
```

## Advantages of Multi-Output Regression

1. **Efficiency**: Single model handles multiple targets
2. **Correlation Capture**: Can capture correlations between targets
3. **Consistency**: Ensures consistent predictions across targets
4. **Simplicity**: Simpler than training separate models
5. **Performance**: Often better than independent models

## When to Use Each Base Estimator

- **Random Forest**: Good all-around performance, interpretable
- **Gradient Boosting**: Best performance, but slower training
- **Linear Regression**: Fast, interpretable, good baseline
- **Ridge Regression**: When you suspect multicollinearity
- **Lasso Regression**: When you want feature selection
- **SVR**: When you need non-linear relationships

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Machine learning algorithms
- matplotlib: Plotting
- seaborn: Statistical visualizations 