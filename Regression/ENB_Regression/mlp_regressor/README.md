# MLP Regressor for Multi-Task Learning on ENB2012 Dataset

This directory contains a comprehensive Multi-Layer Perceptron (MLP) regression framework for multi-task learning on the ENB2012 dataset.

## Overview

The ENB2012 dataset contains building energy efficiency data with:
- **8 input features** (X1-X8): Building characteristics
- **2 target variables** (target_1, target_2): Heating and cooling loads

## Model Features

### MLP Regression Framework (`mlp_regressor.py`)

A comprehensive MLP regression framework that:
- Uses TensorFlow/Keras for deep learning capabilities
- Supports multiple neural network architectures
- Automatically compares different architectures and selects the best performer
- Uses feature scaling for better performance
- Provides detailed evaluation metrics (MSE, MAPE, RMSE)
- Includes cross-validation
- Generates visualizations including loss curves
- Supports early stopping and learning rate scheduling
- Handles multi-task learning with shared layers

## Supported Architectures

The framework tests the following neural network architectures:

1. **Simple MLP**: Basic architecture with 2 hidden layers
2. **Deep MLP**: Deeper architecture with 4 hidden layers
3. **Wide MLP**: Wide architecture with more neurons per layer
4. **Regularized MLP**: Architecture with dropout and L2 regularization
5. **Complex MLP**: Complex architecture with varying layer sizes

## Key Advantages

1. **Architecture Comparison**: Automatically tests multiple neural network architectures
2. **Deep Learning**: Leverages the power of deep neural networks
3. **Regularization**: Built-in dropout and L2 regularization to prevent overfitting
4. **Early Stopping**: Prevents overfitting by monitoring validation loss
5. **Learning Rate Scheduling**: Adaptive learning rate for better convergence
6. **Multi-Task Learning**: Shared layers for correlated targets

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the MLP Regressor Model

```bash
python run_mlp_regressor.py
```

Or directly:

```bash
python mlp_regressor.py
```

### Using the Model Programmatically

```python
from mlp_regressor import MLPRegressor, load_and_preprocess_data

# Load data
X, y, feature_names, target_names = load_and_preprocess_data('../data/ENB2012_data.csv')

# Create and fit model with specific architecture
model = MLPRegressor(
    architecture='deep_mlp',  # or 'simple_mlp', 'wide_mlp', 'regularized_mlp', 'complex_mlp'
    random_state=42
)
model.fit(X, y, feature_names, target_names)

# Make predictions
predictions = model.predict(X)

# Get model summary
model.summary()
```

## Architecture Details

### Simple MLP
- 2 hidden layers: [64, 32]
- ReLU activation
- Basic structure

### Deep MLP
- 4 hidden layers: [128, 64, 32, 16]
- ReLU activation
- Deeper network for complex patterns

### Wide MLP
- 2 hidden layers: [256, 128]
- ReLU activation
- Wider layers for more capacity

### Regularized MLP
- 3 hidden layers: [128, 64, 32]
- ReLU activation
- Dropout (0.3) and L2 regularization (0.01)

### Complex MLP
- 4 hidden layers: [256, 128, 64, 32]
- ReLU activation
- Complex architecture with varying sizes

## Output Files

After running the model, you'll get:

1. **`*_results.csv`**: Detailed performance metrics for the best model
2. **`mlp_architecture_comparison.csv`**: Comparison of all tested architectures
3. **`*_regression_results.png`**: Visualization plots showing true vs predicted values
4. **`*_loss_curves.png`**: Training and validation loss curves

## Performance Metrics

The models evaluate performance using:

- **MSE (Mean Squared Error)**: Average squared difference between predictions and actual values
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **RMSE (Root Mean Squared Error)**: Square root of MSE

## Model Architecture

### MLP Regression Framework

The framework implements a systematic approach:
1. **Data Preprocessing**: Feature scaling using StandardScaler
2. **Architecture Comparison**: Tests multiple neural network architectures
3. **Best Architecture Selection**: Chooses the architecture with lowest MSE
4. **Detailed Analysis**: Performs comprehensive evaluation on the best architecture
5. **Training Visualization**: Plots training and validation loss curves
6. **Cross-Validation**: Validates performance robustness

## Training Configuration

- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Mean Squared Error
- **Batch Size**: 32
- **Epochs**: 200 with early stopping
- **Validation Split**: 20%
- **Early Stopping**: Patience of 20 epochs
- **Learning Rate**: Initial 0.001 with reduction on plateau

## Data Preprocessing

- **Feature Scaling**: StandardScaler is used to normalize features
- **Train-Test Split**: 80-20 split with random state for reproducibility
- **Cross-Validation**: 5-fold cross-validation for robust evaluation

## Architecture Selection Process

The framework follows this process:
1. Tests all supported neural network architectures
2. Evaluates each architecture using MSE, MAPE, and RMSE
3. Selects the best performing architecture by MSE
4. Performs detailed analysis on the selected architecture
5. Generates comprehensive visualizations and reports

## Example Output

```
ARCHITECTURE COMPARISON
============================================================

Testing Simple MLP...
  Average MSE:  8.2345
  Average MAPE: 0.0934
  Average RMSE: 2.8696

Testing Deep MLP...
  Average MSE:  7.5678
  Average MAPE: 0.0856
  Average RMSE: 2.7509

Testing Wide MLP...
  Average MSE:  7.8901
  Average MAPE: 0.0892
  Average RMSE: 2.8090

Best architecture by MSE: Deep MLP (MSE: 7.5678)

MLP DEEP MLP EVALUATION RESULTS
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

## Advantages of MLP Regression

1. **Non-linear Relationships**: Can capture complex non-linear patterns
2. **Feature Learning**: Automatically learns feature representations
3. **Multi-task Learning**: Shared layers can improve performance on correlated targets
4. **Regularization**: Built-in regularization techniques prevent overfitting
5. **Scalability**: Can handle large datasets efficiently

## When to Use Each Architecture

- **Simple MLP**: Good baseline, fast training
- **Deep MLP**: Best for complex patterns, good performance
- **Wide MLP**: When you need more capacity per layer
- **Regularized MLP**: When overfitting is a concern
- **Complex MLP**: When you need maximum capacity

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Data preprocessing and evaluation
- matplotlib: Plotting
- seaborn: Statistical visualizations
- tensorflow: Deep learning framework 