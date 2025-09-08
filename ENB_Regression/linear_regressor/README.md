# Machine Learning Models for ENB2012 Dataset

This directory contains machine learning models for multi-task learning on the ENB2012 dataset.

## Overview

The ENB2012 dataset contains building energy efficiency data with:
- **8 input features** (X1-X8): Building characteristics
- **2 target variables** (target_1, target_2): Heating and cooling loads

## Models Available

### 1. Multi-Task Linear Regression (`linear_regression_mtl.py`)

A comprehensive linear regression model that:
- Handles multiple target variables simultaneously
- Uses feature scaling for better performance
- Provides detailed evaluation metrics (MSE, MAPE, RMSE)
- Includes cross-validation
- Generates visualizations
- Extracts learned equations

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Linear Regression Model

```bash
python run_linear_regression.py
```

Or directly:

```bash
python linear_regression_mtl.py
```

### Using the Model Programmatically

```python
from linear_regression_mtl import MultiTaskLinearRegression, load_and_preprocess_data

# Load data
X, y, feature_names, target_names = load_and_preprocess_data('data/ENB2012_data.csv')

# Create and fit model
model = MultiTaskLinearRegression(random_state=42)
model.fit(X, y, feature_names, target_names)

# Make predictions
predictions = model.predict(X)

# Get model coefficients
coefficients = model.get_coefficients()
intercepts = model.get_intercepts()
```

## Output Files

After running the model, you'll get:

1. **`regression_results.csv`**: Detailed performance metrics for each target
2. **`regression_results.png`**: Visualization plots showing true vs predicted values

## Performance Metrics

The models evaluate performance using:

- **MSE (Mean Squared Error)**: Average squared difference between predictions and actual values
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **RMSE (Root Mean Squared Error)**: Square root of MSE

## Model Architecture

### Multi-Task Linear Regression

The model implements a multi-task learning approach where:
- Each target variable gets its own linear regression model
- Features are standardized using StandardScaler
- All models share the same feature preprocessing
- Results are combined for comprehensive evaluation

## Data Preprocessing

- **Feature Scaling**: StandardScaler is used to normalize features
- **Train-Test Split**: 80-20 split with random state for reproducibility
- **Cross-Validation**: 5-fold cross-validation for robust evaluation

## Example Output

```
MULTI-TASK LINEAR REGRESSION EVALUATION RESULTS
============================================================

TARGET_1:
------------------------------
     MSE: 15.2345
    MAPE: 0.1234
    RMSE: 3.9012

TARGET_2:
------------------------------
     MSE: 18.5678
    MAPE: 0.1456
    RMSE: 4.3090

AVERAGE ACROSS ALL TARGETS
============================================================
     MSE: 16.9012
    MAPE: 0.1345
    RMSE: 4.1051
```

## Extending the Models

To add new models:

1. Create a new Python file in this directory
2. Implement a class with `fit()` and `predict()` methods
3. Add evaluation functions for MSE and MAPE
4. Update this README with documentation

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Machine learning algorithms
- matplotlib: Plotting
- seaborn: Statistical visualizations 