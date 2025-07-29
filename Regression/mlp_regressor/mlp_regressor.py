import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
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

class MLPRegressionModel:
    """
    Multi-layer Perceptron Regression model for predicting multiple target variables.
    """
    
    def __init__(self, hidden_layer_sizes=(100, 50), activation='relu', solver='adam', 
                 alpha=0.0001, learning_rate='adaptive', max_iter=1000, random_state=42):
        """
        Initialize the MLP regression model.
        
        Args:
            hidden_layer_sizes: Tuple of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'logistic')
            solver: Optimizer ('adam', 'sgd', 'lbfgs')
            alpha: L2 regularization parameter
            learning_rate: Learning rate schedule ('constant', 'adaptive', 'invscaling')
            max_iter: Maximum iterations
            random_state: Random seed for reproducibility
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.pipeline = None
        self.feature_names = None
        self.target_names = None
        self.is_fitted = False
        self.training_loss = None
        
        # Initialize the pipeline
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize the pipeline with scaler and MLP regressor."""
        mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        self.pipeline = Pipeline([
            # ('scaler', StandardScaler()),
            ('scaler', MinMaxScaler(feature_range=(0.1, 10.0))),
            ('regressor', mlp)
        ])
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None, 
            target_names: List[str] = None) -> 'MLPRegressionModel':
        """
        Fit the MLP regression model.
        
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
        
        # Get the fitted MLP model for additional information
        mlp_model = self.pipeline.named_steps['regressor']
        
        # Calculate number of parameters
        self.n_parameters_ = sum(coef.size + intercept.size for coef, intercept in zip(mlp_model.coefs_, mlp_model.intercepts_))
        
        # Store training loss history
        self.training_loss = mlp_model.loss_curve_
        
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
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance using the weights of the first hidden layer.
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Use the weights of the first hidden layer as feature importance
        # Take the mean absolute value of weights for each input feature
        mlp_model = self.pipeline.named_steps['regressor']
        first_layer_weights = mlp_model.coefs_[0]  # Shape: (n_features, n_hidden_1)
        feature_importance = np.mean(np.abs(first_layer_weights), axis=1)
        
        return feature_importance
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting model info")
        
        mlp_model = self.pipeline.named_steps['regressor']
        info = {
            'n_layers': len(mlp_model.coefs_) + 1,
            'n_features': mlp_model.n_features_in_,
            'n_outputs': mlp_model.n_outputs_,
            'hidden_layer_sizes': mlp_model.hidden_layer_sizes,
            'activation': mlp_model.activation,
            'solver': mlp_model.solver,
            'alpha': mlp_model.alpha,
            'learning_rate': mlp_model.learning_rate,
            'max_iter': mlp_model.max_iter,
            'n_iterations': mlp_model.n_iter_,
            'final_loss': mlp_model.loss_,
            'converged': mlp_model.n_iter_ < self.max_iter
        }
        
        return info

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

def evaluate_model(model: MLPRegressionModel, X_test: np.ndarray, y_test: np.ndarray, 
                  target_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the model using MSE and MAPE metrics.
    
    Args:
        model: Fitted MLP regression model
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
    print(f"MLP {model_name.upper()} EVALUATION RESULTS")
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
    plt.savefig(f'mlp_regressor/{model_name.lower()}_mlp_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_loss(model: MLPRegressionModel, model_name: str):
    """
    Plot the training loss curve.
    
    Args:
        model: Fitted MLP regression model
        model_name: Name of the model
    """
    if model.training_loss is None:
        print("Training loss not available")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(model.training_loss, 'b-', linewidth=2)
    plt.title(f'Training Loss Curve - {model_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'mlp_regressor/{model_name.lower()}_training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model: MLPRegressionModel, feature_names: List[str], model_name: str):
    """
    Plot feature importance based on first layer weights.
    
    Args:
        model: Fitted MLP regression model
        feature_names: Names of the features
        model_name: Name of the model
    """
    try:
        importance = model.get_feature_importance()
        
        # Sort features by importance
        sorted_idx = np.argsort(importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        plt.figure(figsize=(10, 6))
        plt.barh(pos, importance[sorted_idx])
        plt.yticks(pos)
        plt.yticklabels([feature_names[j] for j in sorted_idx])
        plt.xlabel('Feature Importance (Mean |Weight|)')
        plt.title(f'Feature Importance - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'mlp_regressor/{model_name.lower()}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Feature importance not available: {e}")

def hyperparameter_tuning(X: np.ndarray, y: np.ndarray, target_names: List[str]):
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        X: Features array
        y: Targets array
        target_names: Names of target variables
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Define parameter grid
    param_grid = {
        'regressor__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)],
        'regressor__activation': ['relu', 'tanh'],
        'regressor__alpha': [0.0001, 0.001, 0.01],
        'regressor__learning_rate': ['constant', 'adaptive']
    }
    
    # Create base pipeline
    base_pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        ('scaler', MinMaxScaler(feature_range=(0.1, 10.0))),
        ('regressor', MLPRegressor(
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        ))
    ])
    
    # Perform grid search
    grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X, y)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_params_, grid_search.best_estimator_

def compare_architectures(X: np.ndarray, y: np.ndarray, target_names: List[str]):
    """
    Compare different MLP architectures.
    
    Args:
        X: Features array
        y: Targets array
        target_names: Names of target variables
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define different architectures to test
    architectures = [
        ('Single Layer (50)', (50,)),
        ('Single Layer (100)', (100,)),
        ('Two Layers (50, 25)', (50, 25)),
        ('Two Layers (100, 50)', (100, 50)),
        ('Three Layers (100, 50, 25)', (100, 50, 25)),
        ('Deep Network (100, 75, 50, 25)', (100, 75, 50, 25))
    ]
    
    print("\n" + "="*60)
    print("ARCHITECTURE COMPARISON")
    print("="*60)
    
    comparison_results = {}
    
    for arch_name, hidden_layers in architectures:
        print(f"\nTesting {arch_name}...")
        
        try:
            # Create and fit model
            model = MLPRegressionModel(
                hidden_layer_sizes=hidden_layers,
                max_iter=1000,
                random_state=42
            )
            model.fit(X_train, y_train, feature_names=None, target_names=target_names)
            
            # Evaluate
            results = evaluate_model(model, X_test, y_test, target_names)
            
            # Calculate average metrics
            avg_mse = np.mean([metrics['MSE'] for metrics in results.values()])
            avg_mape = np.mean([metrics['MAPE'] for metrics in results.values()])
            avg_rmse = np.mean([metrics['RMSE'] for metrics in results.values()])
            
            comparison_results[arch_name] = {
                'MSE': avg_mse,
                'MAPE': avg_mape,
                'RMSE': avg_rmse,
                'n_parameters': model.n_parameters_
            }
            
            print(f"  Average MSE:  {avg_mse:.4f}")
            print(f"  Average MAPE: {avg_mape:.4f}")
            print(f"  Average RMSE: {avg_rmse:.4f}")
            print(f"  Parameters:   {model.n_parameters_:,}")
            
        except Exception as e:
            print(f"  Error with {arch_name}: {e}")
            continue
    
    # Find best architecture
    if comparison_results:
        best_arch = min(comparison_results.items(), key=lambda x: x[1]['MSE'])
        print(f"\nBest architecture by MSE: {best_arch[0]} (MSE: {best_arch[1]['MSE']:.4f})")
    
    return comparison_results

def main():
    """
    Main function to run the MLP regression analysis.
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
    
    # Compare different architectures
    comparison_results = compare_architectures(X, y, target_names)
    
    # Use the best architecture for detailed analysis
    best_arch_name = min(comparison_results.items(), key=lambda x: x[1]['MSE'])[0]
    print(f"\nUsing {best_arch_name} for detailed analysis...")
    
    # Extract hidden layer sizes from best architecture name
    if 'Single Layer (50)' in best_arch_name:
        hidden_layers = (50,)
    elif 'Single Layer (100)' in best_arch_name:
        hidden_layers = (100,)
    elif 'Two Layers (50, 25)' in best_arch_name:
        hidden_layers = (50, 25)
    elif 'Two Layers (100, 50)' in best_arch_name:
        hidden_layers = (100, 50)
    elif 'Three Layers (100, 50, 25)' in best_arch_name:
        hidden_layers = (100, 50, 25)
    elif 'Deep Network (100, 75, 50, 25)' in best_arch_name:
        hidden_layers = (100, 75, 50, 25)
    else:
        hidden_layers = (100, 50)  # Default
    
    # Create and fit the best model
    model = MLPRegressionModel(
        hidden_layer_sizes=hidden_layers,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train, feature_names, target_names)
    
    # Print model information
    model_info = model.get_model_info()
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    for key, value in model_info.items():
        print(f"{key:>20}: {value}")
    
    # Evaluate model
    print("\nEvaluating model performance...")
    results = evaluate_model(model, X_test, y_test, target_names)
    print_evaluation_results(results, best_arch_name)
    
    # Create visualizations
    print("\nCreating visualizations...")
    y_pred = model.predict(X_test)
    plot_results(y_test, y_pred, target_names, best_arch_name)
    
    # Plot training loss
    print("\nPlotting training loss...")
    plot_training_loss(model, best_arch_name)
    
    # Plot feature importance
    print("\nPlotting feature importance...")
    plot_feature_importance(model, feature_names, best_arch_name)
    
    # Cross-validation with proper scaling
    print("\nPerforming cross-validation with proper scaling...")
    
    # Create pipeline for cross-validation
    pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        ('scaler', MinMaxScaler(feature_range=(0.1, 10.0))),
        ('regressor', MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        ))
    ])
    
    # MSE cross-validation with proper scaling
    mse_scores = cross_val_score(
        pipeline, X, y, 
        cv=5, scoring='neg_mean_squared_error'
    )
    cv_mse = -mse_scores.mean()
    cv_mse_std = mse_scores.std()
    
    # MAPE cross-validation with proper scaling (using custom scoring)
    def mape_scoring(estimator, X, y):
        y_pred = estimator.predict(X)
        # Calculate MAPE for each target and average
        mape_scores = []
        for i in range(y.shape[1]):
            mape = mean_absolute_percentage_error(y[:, i], y_pred[:, i])
            mape_scores.append(mape)
        return -np.mean(mape_scores)
    
    mape_scores = cross_val_score(
        pipeline, X, y, 
        cv=5, scoring=mape_scoring
    )
    cv_mape = -mape_scores.mean()
    cv_mape_std = mape_scores.std()
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS (5-fold) - WITH PROPER SCALING")
    print("="*60)
    print(f"CV MSE:  {cv_mse:.4f} (+/- {cv_mse_std:.4f})")
    print(f"CV MAPE: {cv_mape:.4f} (+/- {cv_mape_std:.4f})")
    
    # Hyperparameter tuning (optional - can be time-consuming)
    print("\n" + "="*60)
    print("PERFORMING HYPERPARAMETER TUNING")
    print("="*60)
    print("This may take several minutes...")
    
    try:
        best_params, best_estimator = hyperparameter_tuning(X, y, target_names)
        
        # Create final model with best parameters
        final_model = MLPRegressionModel(
            hidden_layer_sizes=best_estimator.named_steps['regressor'].hidden_layer_sizes,
            activation=best_estimator.named_steps['regressor'].activation,
            alpha=best_estimator.named_steps['regressor'].alpha,
            learning_rate=best_estimator.named_steps['regressor'].learning_rate,
            max_iter=1000,
            random_state=42
        )
        final_model.fit(X_train, y_train, feature_names, target_names)
        
        # Evaluate final model
        final_results = evaluate_model(final_model, X_test, y_test, target_names)
        print_evaluation_results(final_results, "Tuned MLP")
        
        # Save final results
        results_df = pd.DataFrame(final_results).T
        results_df.to_csv('mlp_regressor/tuned_mlp_results.csv')
        print(f"\nTuned model results saved to 'mlp_regressor/tuned_mlp_results.csv'")
        
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        print("Using the best architecture model instead.")
    
    # Save results to file
    results_df = pd.DataFrame(results).T
    results_df.to_csv(f'mlp_regressor/{best_arch_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")}_results.csv')
    print(f"\nResults saved to 'mlp_regressor/{best_arch_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}_results.csv'")
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison_results).T
    comparison_df.to_csv('mlp_regressor/mlp_architecture_comparison.csv')
    print(f"Architecture comparison results saved to 'mlp_regressor/mlp_architecture_comparison.csv'")
    
    return model, results, comparison_results

if __name__ == "__main__":
    model, results, comparison_results = main()
