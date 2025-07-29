import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class MultiOutputRegressionModel:
    """
    Multi-output Regression model for predicting multiple target variables.
    """
    
    def __init__(self, base_estimator='random_forest', random_state: int = 42):
        """
        Initialize the multi-output regression model.
        
        Args:
            base_estimator: Base estimator to use ('random_forest', 'gradient_boosting', 'linear', 'ridge', 'lasso', 'svr')
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.base_estimator_name = base_estimator
        self.pipeline = None
        self.feature_names = None
        self.target_names = None
        self.is_fitted = False
        
        # Initialize pipeline
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize the pipeline with scaler and base estimator."""
        if self.base_estimator_name == 'random_forest':
            base_est = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            )
        elif self.base_estimator_name == 'gradient_boosting':
            base_est = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            )
        elif self.base_estimator_name == 'linear':
            base_est = LinearRegression()
        elif self.base_estimator_name == 'ridge':
            base_est = Ridge(alpha=1.0, random_state=self.random_state)
        elif self.base_estimator_name == 'lasso':
            base_est = Lasso(alpha=0.1, random_state=self.random_state)
        elif self.base_estimator_name == 'svr':
            base_est = SVR(kernel='rbf', C=1.0, gamma='scale')
        else:
            raise ValueError(f"Unknown base estimator: {self.base_estimator_name}")
        
        # Create pipeline with scaler and multi-output regressor
        self.pipeline = Pipeline([
            # ('scaler', StandardScaler()),
            ('scaler', MinMaxScaler(feature_range=(0.1, 10.0))),
            ('regressor', MultiOutputRegressor(base_est))
        ])
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None, 
            target_names: List[str] = None) -> 'MultiOutputRegressionModel':
        """
        Fit the multi-output regression model.
        
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
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
        
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
            
        return self.pipeline.predict(X)
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance for each target variable (if available).
        
        Returns:
            Dictionary mapping target names to their feature importance arrays
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        regressor = self.pipeline.named_steps['regressor']
        if not hasattr(regressor.estimators_[0], 'feature_importances_'):
            raise ValueError("Base estimator does not support feature importance")
            
        importance_dict = {}
        for i, target_name in enumerate(self.target_names):
            importance_dict[target_name] = regressor.estimators_[i].feature_importances_
            
        return importance_dict
    
    def get_coefficients(self) -> Dict[str, np.ndarray]:
        """
        Get coefficients for each target variable (for linear models).
        
        Returns:
            Dictionary mapping target names to their coefficient arrays
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")
        
        regressor = self.pipeline.named_steps['regressor']
        if not hasattr(regressor.estimators_[0], 'coef_'):
            raise ValueError("Base estimator does not support coefficients")
            
        coefficients_dict = {}
        for i, target_name in enumerate(self.target_names):
            coefficients_dict[target_name] = regressor.estimators_[i].coef_
            
        return coefficients_dict

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

def evaluate_model(model: MultiOutputRegressionModel, X_test: np.ndarray, y_test: np.ndarray, 
                  target_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the model using MSE and MAPE metrics.
    
    Args:
        model: Fitted multi-output regression model
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

def print_evaluation_results(results: Dict[str, Dict[str, float]], model_name: str):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Evaluation results dictionary
        model_name: Name of the model
    """
    print("\n" + "="*60)
    print(f"MULTI-OUTPUT {model_name.upper()} EVALUATION RESULTS")
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

def plot_results(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str], model_name: str):
    """
    Create visualization plots for the results.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        target_names: Names of target variables
        model_name: Name of the model
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
        ax.set_title(f'{target_name}: True vs Predicted ({model_name})')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'multiout_regressor/{model_name.lower()}_regression_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model: MultiOutputRegressionModel, feature_names: List[str], target_names: List[str]):
    """
    Plot feature importance for each target variable.
    
    Args:
        model: Fitted multi-output regression model
        feature_names: Names of the features
        target_names: Names of target variables
    """
    try:
        importance_dict = model.get_feature_importance()
        
        fig, axes = plt.subplots(1, len(target_names), figsize=(15, 5))
        if len(target_names) == 1:
            axes = [axes]
        
        for i, target_name in enumerate(target_names):
            ax = axes[i]
            importance = importance_dict[target_name]
            
            # Sort features by importance
            sorted_idx = np.argsort(importance)
            pos = np.arange(sorted_idx.shape[0]) + .5
            
            ax.barh(pos, importance[sorted_idx])
            ax.set_yticks(pos)
            ax.set_yticklabels([feature_names[j] for j in sorted_idx])
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Feature Importance for {target_name}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multiout_regressor/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ValueError as e:
        print(f"Feature importance not available: {e}")

def compare_models(X: np.ndarray, y: np.ndarray, target_names: List[str]):
    """
    Compare different multi-output regression models.
    
    Args:
        X: Features array
        y: Targets array
        target_names: Names of target variables
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define models to compare
    models_to_test = [
        ('Random Forest', 'random_forest'),
        ('Gradient Boosting', 'gradient_boosting'),
        ('Linear Regression', 'linear'),
        ('Ridge Regression', 'ridge'),
        ('Lasso Regression', 'lasso')
    ]
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_results = {}
    
    for model_name, estimator_type in models_to_test:
        print(f"\nTesting {model_name}...")
        
        try:
            # Create and fit model
            model = MultiOutputRegressionModel(
                base_estimator=estimator_type,
                random_state=42
            )
            model.fit(X_train, y_train, feature_names=None, target_names=target_names)
            
            # Evaluate
            results = evaluate_model(model, X_test, y_test, target_names)
            
            # Calculate average metrics
            avg_mse = np.mean([metrics['MSE'] for metrics in results.values()])
            avg_mape = np.mean([metrics['MAPE'] for metrics in results.values()])
            avg_rmse = np.mean([metrics['RMSE'] for metrics in results.values()])
            
            comparison_results[model_name] = {
                'MSE': avg_mse,
                'MAPE': avg_mape,
                'RMSE': avg_rmse
            }
            
            print(f"  Average MSE:  {avg_mse:.4f}")
            print(f"  Average MAPE: {avg_mape:.4f}")
            print(f"  Average RMSE: {avg_rmse:.4f}")
            
        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            continue
    
    # Find best model
    if comparison_results:
        best_model = min(comparison_results.items(), key=lambda x: x[1]['MSE'])
        print(f"\nBest model by MSE: {best_model[0]} (MSE: {best_model[1]['MSE']:.4f})")
    
    return comparison_results

def main():
    """
    Main function to run the multi-output regression analysis.
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
    
    # Compare different models
    print("\nComparing different multi-output regression models...")
    comparison_results = compare_models(X, y, target_names)
    
    # Use the best model for detailed analysis
    best_model_name = min(comparison_results.items(), key=lambda x: x[1]['MSE'])[0]
    print(f"\nUsing {best_model_name} for detailed analysis...")
    
    # Map display name to internal estimator name
    name_mapping = {
        'Random Forest': 'random_forest',
        'Gradient Boosting': 'gradient_boosting',
        'Linear Regression': 'linear',
        'Ridge Regression': 'ridge',
        'Lasso Regression': 'lasso'
    }
    
    estimator_type = name_mapping.get(best_model_name, 'random_forest')
    
    # Create and fit the best model
    model = MultiOutputRegressionModel(base_estimator=estimator_type, random_state=42)
    model.fit(X_train, y_train, feature_names, target_names)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    results = evaluate_model(model, X_test, y_test, target_names)
    print_evaluation_results(results, best_model_name)
    
    # Create visualizations
    print("\nCreating visualizations...")
    y_pred = model.predict(X_test)
    plot_results(y_test, y_pred, target_names, best_model_name)
    
    # Plot feature importance if available
    try:
        print("\nPlotting feature importance...")
        plot_feature_importance(model, feature_names, target_names)
    except Exception as e:
        print(f"Feature importance plotting failed: {e}")
    
    # Cross-validation with proper scaling
    print("\nPerforming cross-validation with proper scaling...")
    
    # Create pipeline for cross-validation
    if estimator_type == 'random_forest':
        base_est = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    elif estimator_type == 'gradient_boosting':
        base_est = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    elif estimator_type == 'linear':
        base_est = LinearRegression()
    elif estimator_type == 'ridge':
        base_est = Ridge(alpha=1.0, random_state=42)
    elif estimator_type == 'lasso':
        base_est = Lasso(alpha=0.1, random_state=42)
    elif estimator_type == 'svr':
        base_est = SVR(kernel='rbf', C=1.0, gamma='scale')
    
    pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        ('scaler', MinMaxScaler(feature_range=(0.1, 10.0))),
        ('regressor', MultiOutputRegressor(base_est))
    ])
    
    # MSE cross-validation with proper scaling
    mse_scores = cross_val_score(
        pipeline, X, y, 
        cv=5, scoring='neg_mean_squared_error'
    )
    cv_mse = -mse_scores.mean()
    cv_mse_std = mse_scores.std()
    
    # MAPE cross-validation with proper scaling (using custom scoring)
    def mape_scoring_multi(estimator, X, y):
        y_pred = estimator.predict(X)
        # Calculate MAPE for each target and average
        mape_scores = []
        for i in range(y.shape[1]):
            mape = mean_absolute_percentage_error(y[:, i], y_pred[:, i])
            mape_scores.append(mape)
        return -np.mean(mape_scores)
    
    mape_scores = cross_val_score(
        pipeline, X, y, 
        cv=5, scoring=mape_scoring_multi
    )
    cv_mape = -mape_scores.mean()
    cv_mape_std = mape_scores.std()
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS (5-fold) - WITH PROPER SCALING")
    print("="*60)
    print(f"CV MSE:  {cv_mse:.4f} (+/- {cv_mse_std:.4f})")
    print(f"CV MAPE: {cv_mape:.4f} (+/- {cv_mape_std:.4f})")
    
    # Save results to file
    results_df = pd.DataFrame(results).T
    results_df.to_csv(f'multiout_regressor/{best_model_name}_regression_results.csv')
    print(f"\nResults saved to 'multiout_regressor/{best_model_name}_regression_results.csv'")
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison_results).T
    comparison_df.to_csv('multiout_regressor/model_comparison_results.csv')
    print(f"Model comparison results saved to 'multiout_regressor/model_comparison_results.csv'")
    
    return model, results, comparison_results

if __name__ == "__main__":
    model, results, comparison_results = main()
