# SGD Regressor for Multi-Task Learning on ENB2012 Dataset

This directory contains a Stochastic Gradient Descent (SGD) Regressor model for multi-task learning on the ENB2012 dataset.

## Overview

The ENB2012 dataset contains building energy efficiency data with:
- **8 input features** (X1-X8): Building characteristics
- **2 target variables** (target_1, target_2): Heating and cooling loads

## Model Features

### Multi-Task SGD Regressor (`sgd_regressor.py`)

A comprehensive SGD regressor model that:
- Handles multiple target variables simultaneously
- Uses feature scaling for better performance
- Implements adaptive learning rate and early stopping
- Provides detailed evaluation metrics (MSE, MAPE, RMSE)
- Includes cross-validation
- Compares performance with standard Linear Regression
- Generates visualizations
- Extracts learned equations

## Key Advantages of SGD Regressor

1. **Scalability**: Efficient for large datasets
2. **Online Learning**: Can handle streaming data
3. **Regularization**: Built-in support for L1/L2 regularization
4. **Adaptive Learning**: Automatic learning rate adjustment
5. **Early Stopping**: Prevents overfitting

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the SGD Regressor Model

```bash
python run_sgd_regressor.py
```

Or directly:

```bash
python sgd_regressor.py
```

### Using the Model Programmatically

```python
from sgd_regressor import MultiTaskSGDRegressor, load_and_preprocess_data

# Load data
X, y, feature_names, target_names = load_and_preprocess_data('../data/ENB2012_data.csv')

# Create and fit model
model = MultiTaskSGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
model.fit(X, y, feature_names, target_names)

# Make predictions
predictions = model.predict(X)

# Get model coefficients
coefficients = model.get_coefficients()
intercepts = model.get_intercepts()
```

## Model Parameters

The SGD Regressor uses the following key parameters:
- `max_iter=1000`: Maximum iterations for convergence
- `tol=1e-3`: Tolerance for convergence
- `learning_rate='adaptive'`: Automatic learning rate adjustment
- `early_stopping=True`: Stop training when validation score doesn't improve
- `validation_fraction=0.1`: Fraction of training data for validation

## Output Files

After running the model, you'll get:

1. **`sgd_regression_results.csv`**: Detailed performance metrics for each target
2. **`sgd_vs_linear_comparison.csv`**: Comparison with standard Linear Regression
3. **`sgd_regression_results.png`**: Visualization plots showing true vs predicted values

## Performance Metrics

The models evaluate performance using:

- **MSE (Mean Squared Error)**: Average squared difference between predictions and actual values
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **RMSE (Root Mean Squared Error)**: Square root of MSE

## Model Architecture

### Multi-Task SGD Regressor

The model implements a multi-task learning approach where:
- Each target variable gets its own SGD regressor model
- Features are standardized using StandardScaler
- All models share the same feature preprocessing
- Results are combined for comprehensive evaluation
- Performance is compared with standard Linear Regression

## Data Preprocessing

- **Feature Scaling**: StandardScaler is used to normalize features
- **Train-Test Split**: 80-20 split with random state for reproducibility
- **Cross-Validation**: 5-fold cross-validation for robust evaluation

## Comparison with Linear Regression

The model automatically compares SGD performance with standard Linear Regression to show:
- Which method performs better for each target
- Trade-offs between computational efficiency and accuracy
- Benefits of SGD's adaptive learning

## Example Output

```
MULTI-TASK SGD REGRESSOR EVALUATION RESULTS
============================================================

TARGET_1:
------------------------------
     MSE: 14.2345
    MAPE: 0.1134
    RMSE: 3.7728

TARGET_2:
------------------------------
     MSE: 17.5678
    MAPE: 0.1356
    RMSE: 4.1914

AVERAGE ACROSS ALL TARGETS
============================================================
     MSE: 15.9012
    MAPE: 0.1245
    RMSE: 3.9821

COMPARISON: SGD vs LINEAR REGRESSION
============================================================

TARGET_1:
------------------------------
SGD Regressor:
  MSE:  14.2345
  MAPE: 0.1134
Linear Regression:
  MSE:  15.1234
  MAPE: 0.1234
```

## Advantages of SGD over Linear Regression

1. **Memory Efficiency**: Processes data in batches
2. **Online Learning**: Can update with new data
3. **Regularization**: Built-in L1/L2 regularization
4. **Adaptive Learning**: Automatic learning rate adjustment
5. **Scalability**: Better for large datasets

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Machine learning algorithms
- matplotlib: Plotting
- seaborn: Statistical visualizations 