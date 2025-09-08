# Gaussian Process Regressor for Multi-Task Learning on ENB2012 Dataset

This directory contains a Gaussian Process (GP) Regressor for multi-task learning on the ENB2012 dataset.

## Overview

The ENB2012 dataset contains building energy efficiency data with:
- **8 input features** (X1-X8): Building characteristics
- **2 target variables** (target_1, target_2): Heating and cooling loads

## Model Features

### Multi-Task Gaussian Process Regressor (`gp_regressor.py`)

A comprehensive Gaussian Process regressor model that:
- Handles multiple target variables simultaneously
- Uses feature scaling for better performance
- Automatically optimizes kernel hyperparameters
- Provides uncertainty quantification (standard deviations)
- Tests multiple kernel configurations (RBF, Matern, combinations)
- Provides detailed evaluation metrics (MSE, MAPE, RMSE, Mean_Std)
- Includes cross-validation
- Generates visualizations with uncertainty bands
- Extracts optimized kernel information and log marginal likelihood

## Key Advantages of Gaussian Processes

1. **Uncertainty Quantification**: Provides prediction uncertainty estimates
2. **Non-parametric**: No assumptions about the underlying function form
3. **Kernel Flexibility**: Can model various types of relationships
4. **Probabilistic**: Full posterior distribution over predictions
5. **Automatic Hyperparameter Optimization**: Learns optimal kernel parameters

## Kernel Types Tested

The model automatically tests and selects the best kernel from:
- **RBF + White**: Radial Basis Function with noise
- **Matern + White**: Matern kernel with noise (better for non-smooth functions)
- **RBF + Matern + White**: Combination of both kernels

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Gaussian Process Regressor Model

```bash
python run_gp_regressor.py
```

Or directly:

```bash
python gp_regressor.py
```

### Using the Model Programmatically

```python
from gp_regressor import MultiTaskGaussianProcess, load_and_preprocess_data
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

# Load data
X, y, feature_names, target_names = load_and_preprocess_data('../data/ENB2012_data.csv')

# Create custom kernel
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

# Create and fit model
model = MultiTaskGaussianProcess(
    kernel=kernel,
    random_state=42,
    n_restarts_optimizer=10
)
model.fit(X, y, feature_names, target_names)

# Make predictions with uncertainty
predictions, stds = model.predict(X, return_std=True)

# Get optimized kernels
kernels = model.get_kernels()
log_likelihoods = model.get_log_marginal_likelihood()
```

## Model Parameters

The Gaussian Process Regressor uses the following key parameters:
- `n_restarts_optimizer=10`: Number of restarts for kernel optimization
- `normalize_y=True`: Normalize target values
- `random_state=42`: For reproducibility

## Output Files

After running the model, you'll get:

1. **`gp_regression_results.csv`**: Detailed performance metrics for each target
2. **`gp_regression_results.png`**: Visualization plots showing true vs predicted values with uncertainty bands

## Performance Metrics

The models evaluate performance using:

- **MSE (Mean Squared Error)**: Average squared difference between predictions and actual values
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **RMSE (Root Mean Squared Error)**: Square root of MSE
- **Mean_Std**: Average prediction uncertainty (standard deviation)

## Model Architecture

### Multi-Task Gaussian Process Regressor

The model implements a multi-task learning approach where:
- Each target variable gets its own Gaussian Process model
- Features are standardized using StandardScaler
- All models share the same feature preprocessing
- Kernel hyperparameters are optimized for each target
- Results include both predictions and uncertainty estimates

## Data Preprocessing

- **Feature Scaling**: StandardScaler is used to normalize features
- **Train-Test Split**: 80-20 split with random state for reproducibility
- **Cross-Validation**: 5-fold cross-validation for robust evaluation

## Uncertainty Quantification

One of the key advantages of Gaussian Processes is uncertainty quantification:
- **Standard Deviations**: Each prediction comes with an uncertainty estimate
- **Confidence Intervals**: Can construct confidence intervals around predictions
- **Model Uncertainty**: Captures both data noise and model uncertainty

## Kernel Optimization

The model automatically:
1. Tests multiple kernel configurations
2. Optimizes hyperparameters using maximum likelihood
3. Selects the best performing kernel
4. Reports optimized kernel parameters and log marginal likelihood

## Example Output

```
MULTI-TASK GAUSSIAN PROCESS EVALUATION RESULTS
============================================================

TARGET_1:
------------------------------
       MSE: 8.2345
      MAPE: 0.0934
      RMSE: 2.8696
   Mean_Std: 1.2345

TARGET_2:
------------------------------
       MSE: 9.5678
      MAPE: 0.1056
      RMSE: 3.0934
   Mean_Std: 1.3456

AVERAGE ACROSS ALL TARGETS
============================================================
       MSE: 8.9012
      MAPE: 0.0995
      RMSE: 2.9815
   Mean_Std: 1.2901

OPTIMIZED KERNELS AND MODEL INFORMATION
============================================================

TARGET_1:
------------------------------
Optimized Kernel: 1.41**2 * RBF(length_scale=0.85) + WhiteKernel(noise_level=0.1)
Log Marginal Likelihood: -245.6789

TARGET_2:
------------------------------
Optimized Kernel: 1.23**2 * Matern(length_scale=0.92, nu=1.5) + WhiteKernel(noise_level=0.15)
Log Marginal Likelihood: -267.8901
```

## Advantages of Gaussian Processes

1. **Uncertainty Quantification**: Provides prediction confidence intervals
2. **Non-parametric**: No assumptions about function form
3. **Kernel Flexibility**: Can model complex relationships
4. **Automatic Complexity Control**: Built-in regularization
5. **Probabilistic Output**: Full posterior distributions

## Computational Considerations

- **Training Time**: Kernel optimization can be computationally expensive
- **Memory Usage**: Scales with O(n²) for n training points
- **Prediction Time**: Scales with O(n) for each prediction
- **Hyperparameter Optimization**: Multiple restarts increase training time

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Machine learning algorithms and Gaussian Processes
- matplotlib: Plotting
- seaborn: Statistical visualizations
- scipy: Scientific computing (required for GP optimization) 