import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class MultiTaskSGDRegressor:
    """
    Multi-task SGD Regressor model for predicting multiple target variables.
    """
    
    def __init__(self, random_state: int = 42, max_iter: int = 1000, tol: float = 1e-3):
        """
        Initialize the multi-task SGD regressor model.
        
        Args:
            random_state: Random seed for reproducibility
            max_iter: Maximum number of iterations for SGD
            tol: Tolerance for convergence
        """
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.pipelines = {}
        self.feature_names = None
        self.target_names = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None, 
            target_names: List[str] = None) -> 'MultiTaskSGDRegressor':
        """
        Fit the multi-task SGD regressor model.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target variables (n_samples, n_targets)
            feature_names: Names of the features
            target_names: Names of the target variables
            
        Returns:
            self: Fitted model
        """
        self.feature_names = feature_names
        self.target_names = target_names
        
        # Fit separate pipeline for each target
        for i in range(y.shape[1]):
            target_name = target_names[i] if target_names else f'target_{i+1}'
            
            # Create pipeline with scaler and regressor
            pipeline = Pipeline([
                # ('scaler', StandardScaler()),
                ('scaler', MinMaxScaler(feature_range=(0.1, 10.0))),
                ('regressor', SGDRegressor(
                    random_state=self.random_state,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    learning_rate='adaptive',
                    early_stopping=True,
                    validation_fraction=0.1
                ))
            ])
            
            pipeline.fit(X, y[:, i])
            self.pipelines[target_name] = pipeline
            
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target variables for given input features.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples, n_targets)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        predictions = []
        
        for target_name in self.pipelines.keys():
            pred = self.pipelines[target_name].predict(X)
            predictions.append(pred)
            
        return np.column_stack(predictions)
    
    def get_coefficients(self) -> Dict[str, np.ndarray]:
        """
        Get the coefficients for each target variable.
        
        Returns:
            Dictionary mapping target names to their coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")
            
        coefficients = {}
        for target_name, pipeline in self.pipelines.items():
            coefficients[target_name] = pipeline.named_steps['regressor'].coef_
            
        return coefficients
    
    def get_intercepts(self) -> Dict[str, float]:
        """
        Get the intercepts for each target variable.
        
        Returns:
            Dictionary mapping target names to their intercepts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting intercepts")
            
        intercepts = {}
        for target_name, pipeline in self.pipelines.items():
            intercepts[target_name] = pipeline.named_steps['regressor'].intercept_
            
        return intercepts

def load_and_preprocess_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load and preprocess the ENB2012 dataset.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        X: Features array
        y: Targets array
        feature_names: List of feature names
        target_names: List of target names
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Separate features and targets
    feature_cols = [col for col in df.columns if col.startswith('X')]
    target_cols = [col for col in df.columns if col.startswith('target')]
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    feature_names = feature_cols
    target_names = target_cols
    
    return X, y, feature_names, target_names

def evaluate_model(model: MultiTaskSGDRegressor, X_test: np.ndarray, y_test: np.ndarray, 
                  target_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the model using MSE and MAPE metrics.
    
    Args:
        model: Fitted multi-task SGD regressor model
        X_test: Test features
        y_test: Test targets
        target_names: Names of target variables
        
    Returns:
        Dictionary containing evaluation metrics for each target
    """
    y_pred = model.predict(X_test)
    
    results = {}
    
    for i, target_name in enumerate(target_names):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        mape = mean_absolute_percentage_error(y_test[:, i], y_pred[:, i])
        
        results[target_name] = {
            'MSE': mse,
            'MAPE': mape,
            'RMSE': np.sqrt(mse)
        }
    
    return results

def print_evaluation_results(results: Dict[str, Dict[str, float]]):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("MULTI-TASK SGD REGRESSOR EVALUATION RESULTS")
    print("="*60)
    
    for target_name, metrics in results.items():
        print(f"\n{target_name.upper()}:")
        print("-" * 30)
        for metric_name, value in metrics.items():
            print(f"{metric_name:>8}: {value:.4f}")
    
    # Calculate average metrics across all targets
    avg_mse = np.mean([metrics['MSE'] for metrics in results.values()])
    avg_mape = np.mean([metrics['MAPE'] for metrics in results.values()])
    avg_rmse = np.mean([metrics['RMSE'] for metrics in results.values()])
    
    print(f"\n{'AVERAGE ACROSS ALL TARGETS':^60}")
    print("-" * 60)
    print(f"{'MSE':>8}: {avg_mse:.4f}")
    print(f"{'MAPE':>8}: {avg_mape:.4f}")
    print(f"{'RMSE':>8}: {avg_rmse:.4f}")

def plot_results(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]):
    """
    Create visualization plots for the results.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        target_names: Names of target variables
    """
    fig, axes = plt.subplots(1, len(target_names), figsize=(15, 5))
    if len(target_names) == 1:
        axes = [axes]
    
    for i, target_name in enumerate(target_names):
        ax = axes[i]
        
        # Scatter plot of true vs predicted
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
        
        # Perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_xlabel(f'True {target_name}')
        ax.set_ylabel(f'Predicted {target_name}')
        ax.set_title(f'{target_name}: True vs Predicted (SGD)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sgd_regressor/sgd_regression_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_model_equations(model: MultiTaskSGDRegressor, feature_names: List[str], 
                         target_names: List[str]):
    """
    Print the learned linear equations for each target.
    
    Args:
        model: Fitted multi-task SGD regressor model
        feature_names: Names of the features
        target_names: Names of the target variables
    """
    coefficients = model.get_coefficients()
    intercepts = model.get_intercepts()
    
    print("\n" + "="*60)
    print("LEARNED LINEAR EQUATIONS (SGD)")
    print("="*60)
    
    for target_name in target_names:
        print(f"\n{target_name.upper()} = {float(intercepts[target_name]):.4f}")
        
        for i, feature_name in enumerate(feature_names):
            coef = coefficients[target_name][i]
            if coef >= 0:
                print(f"         + {coef:.4f} × {feature_name}")
            else:
                print(f"         - {abs(coef):.4f} × {feature_name}")

def compare_with_linear_regression(X: np.ndarray, y: np.ndarray, target_names: List[str]):
    """
    Compare SGD performance with standard Linear Regression.
    
    Args:
        X: Features array
        y: Targets array
        target_names: Names of target variables
    """
    from sklearn.linear_model import LinearRegression
    
    print("\n" + "="*60)
    print("COMPARISON: SGD vs LINEAR REGRESSION")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(0.1, 10.0))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results_comparison = {}
    
    for i, target_name in enumerate(target_names):
        print(f"\n{target_name.upper()}:")
        print("-" * 30)
        
        # SGD Regressor
        sgd_model = SGDRegressor(
            random_state=42,
            max_iter=1000,
            tol=1e-3,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.1
        )
        sgd_model.fit(X_train_scaled, y_train[:, i])
        sgd_pred = sgd_model.predict(X_test_scaled)
        sgd_mse = mean_squared_error(y_test[:, i], sgd_pred)
        sgd_mape = mean_absolute_percentage_error(y_test[:, i], sgd_pred)
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train[:, i])
        lr_pred = lr_model.predict(X_test_scaled)
        lr_mse = mean_squared_error(y_test[:, i], lr_pred)
        lr_mape = mean_absolute_percentage_error(y_test[:, i], lr_pred)
        
        print(f"SGD Regressor:")
        print(f"  MSE:  {sgd_mse:.4f}")
        print(f"  MAPE: {sgd_mape:.4f}")
        print(f"Linear Regression:")
        print(f"  MSE:  {lr_mse:.4f}")
        print(f"  MAPE: {lr_mape:.4f}")
        
        results_comparison[target_name] = {
            'SGD_MSE': sgd_mse,
            'SGD_MAPE': sgd_mape,
            'LR_MSE': lr_mse,
            'LR_MAPE': lr_mape
        }
    
    return results_comparison

def main():
    """
    Main function to run the multi-task SGD regressor analysis.
    """
    print("Loading and preprocessing ENB2012 dataset...")
    
    # Load data
    X, y, feature_names, target_names = load_and_preprocess_data('data/ENB2012_data.csv')
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Targets: {target_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and fit model
    print("\nTraining multi-task SGD regressor model...")
    model = MultiTaskSGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train, feature_names, target_names)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    results = evaluate_model(model, X_test, y_test, target_names)
    print_evaluation_results(results)
    
    # Print model equations
    print_model_equations(model, feature_names, target_names)
    
    # Create visualizations
    print("\nCreating visualizations...")
    y_pred = model.predict(X_test)
    plot_results(y_test, y_pred, target_names)
    
    # Compare with linear regression
    print("\nComparing with linear regression...")
    comparison_results = compare_with_linear_regression(X, y, target_names)
    
    # Cross-validation with proper scaling
    print("\nPerforming cross-validation with proper scaling...")
    cv_scores_mse = {}
    cv_scores_mape = {}
    
    for i, target_name in enumerate(target_names):
        # Create pipeline for this target
        pipeline = Pipeline([
            # ('scaler', StandardScaler()),
            ('scaler', MinMaxScaler(feature_range=(0.1, 10.0))),
            ('regressor', SGDRegressor(
                random_state=42,
                max_iter=1000,
                tol=1e-3,
                learning_rate='adaptive',
                early_stopping=True,
                validation_fraction=0.1
            ))
        ])
        
        # MSE cross-validation with proper scaling
        mse_scores = cross_val_score(
            pipeline, X, y[:, i], 
            cv=5, scoring='neg_mean_squared_error'
        )
        cv_scores_mse[target_name] = -mse_scores.mean()
        
        # MAPE cross-validation with proper scaling (using custom scoring)
        def mape_scoring(estimator, X, y):
            y_pred = estimator.predict(X)
            return -mean_absolute_percentage_error(y, y_pred)
        
        mape_scores = cross_val_score(
            pipeline, X, y[:, i], 
            cv=5, scoring=mape_scoring
        )
        cv_scores_mape[target_name] = -mape_scores.mean()
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS (5-fold) - WITH PROPER SCALING")
    print("="*60)
    
    for target_name in target_names:
        print(f"\n{target_name.upper()}:")
        print(f"  CV MSE:  {cv_scores_mse[target_name]:.4f}")
        print(f"  CV MAPE: {cv_scores_mape[target_name]:.4f}")
    
    # Save results to file
    results_df = pd.DataFrame(results).T
    results_df.to_csv('sgd_regressor/sgd_regression_results.csv')
    print(f"\nResults saved to 'sgd_regressor/sgd_regression_results.csv'")
    
    return model, results

if __name__ == "__main__":
    model, results = main()
