import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
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

class MultiTaskGaussianProcess:
    """
    Multi-task Gaussian Process Regressor for predicting multiple target variables.
    """
    
    def __init__(self, kernel=None, random_state: int = 42, n_restarts_optimizer: int = 10):
        """
        Initialize the multi-task Gaussian Process regressor.
        
        Args:
            kernel: Kernel function for the GP (default: RBF + WhiteKernel)
            random_state: Random seed for reproducibility
            n_restarts_optimizer: Number of restarts for kernel optimization
        """
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer
        self.pipelines = {}
        self.feature_names = None
        self.target_names = None
        self.is_fitted = False
        
        # Default kernel: RBF + WhiteKernel
        if kernel is None:
            self.kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        else:
            self.kernel = kernel
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None, 
            target_names: List[str] = None) -> 'MultiTaskGaussianProcess':
        """
        Fit the multi-task Gaussian Process regressor.
        
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
            
            # Create kernel for this target (copy to avoid sharing)
            if hasattr(self.kernel, 'clone_with_theta'):
                kernel = self.kernel.clone_with_theta(self.kernel.theta)
            else:
                kernel = self.kernel
            
            # Create pipeline with scaler and GP regressor
            pipeline = Pipeline([
                # ('scaler', StandardScaler()),
                ('scaler', MinMaxScaler(feature_range=(0.1, 10.0))),
                ('regressor', GaussianProcessRegressor(
                    kernel=kernel,
                    random_state=self.random_state,
                    n_restarts_optimizer=self.n_restarts_optimizer,
                    normalize_y=True
                ))
            ])
            
            pipeline.fit(X, y[:, i])
            self.pipelines[target_name] = pipeline
            
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict target variables for given input features.
        
        Args:
            X: Input features (n_samples, n_features)
            return_std: Whether to return standard deviations
            
        Returns:
            Predictions (n_samples, n_targets) and optionally standard deviations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        predictions = []
        stds = []
        
        for target_name in self.pipelines.keys():
            if return_std:
                pred, std = self.pipelines[target_name].predict(X, return_std=True)
                predictions.append(pred)
                stds.append(std)
            else:
                pred = self.pipelines[target_name].predict(X)
                predictions.append(pred)
        
        if return_std:
            return np.column_stack(predictions), np.column_stack(stds)
        else:
            return np.column_stack(predictions)
    
    def get_kernels(self) -> Dict[str, object]:
        """
        Get the optimized kernels for each target variable.
        
        Returns:
            Dictionary mapping target names to their optimized kernels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting kernels")
            
        kernels = {}
        for target_name, pipeline in self.pipelines.items():
            kernels[target_name] = pipeline.named_steps['regressor'].kernel_
            
        return kernels
    
    def get_log_marginal_likelihood(self) -> Dict[str, float]:
        """
        Get the log marginal likelihood for each target variable.
        
        Returns:
            Dictionary mapping target names to their log marginal likelihood
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting log marginal likelihood")
            
        log_likelihoods = {}
        for target_name, pipeline in self.pipelines.items():
            log_likelihoods[target_name] = pipeline.named_steps['regressor'].log_marginal_likelihood()
            
        return log_likelihoods

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

def evaluate_model(model: MultiTaskGaussianProcess, X_test: np.ndarray, y_test: np.ndarray, 
                  target_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the model using MSE and MAPE metrics.
    
    Args:
        model: Fitted multi-task Gaussian Process model
        X_test: Test features
        y_test: Test targets
        target_names: Names of target variables
        
    Returns:
        Dictionary containing evaluation metrics for each target
    """
    y_pred, y_std = model.predict(X_test, return_std=True)
    
    results = {}
    
    for i, target_name in enumerate(target_names):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        mape = mean_absolute_percentage_error(y_test[:, i], y_pred[:, i])
        
        results[target_name] = {
            'MSE': mse,
            'MAPE': mape,
            'RMSE': np.sqrt(mse),
            'Mean_Std': np.mean(y_std[:, i])
        }
    
    return results

def print_evaluation_results(results: Dict[str, Dict[str, float]]):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("MULTI-TASK GAUSSIAN PROCESS EVALUATION RESULTS")
    print("="*60)
    
    for target_name, metrics in results.items():
        print(f"\n{target_name.upper()}:")
        print("-" * 30)
        for metric_name, value in metrics.items():
            print(f"{metric_name:>10}: {value:.4f}")
    
    # Calculate average metrics across all targets
    avg_mse = np.mean([metrics['MSE'] for metrics in results.values()])
    avg_mape = np.mean([metrics['MAPE'] for metrics in results.values()])
    avg_rmse = np.mean([metrics['RMSE'] for metrics in results.values()])
    avg_std = np.mean([metrics['Mean_Std'] for metrics in results.values()])
    
    print(f"\n{'AVERAGE ACROSS ALL TARGETS':^60}")
    print("-" * 60)
    print(f"{'MSE':>10}: {avg_mse:.4f}")
    print(f"{'MAPE':>10}: {avg_mape:.4f}")
    print(f"{'RMSE':>10}: {avg_rmse:.4f}")
    print(f"{'Mean_Std':>10}: {avg_std:.4f}")

def plot_results(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, target_names: List[str]):
    """
    Create visualization plots for the results.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        y_std: Standard deviations of predictions
        target_names: Names of target variables
    """
    fig, axes = plt.subplots(1, len(target_names), figsize=(15, 5))
    if len(target_names) == 1:
        axes = [axes]
    
    for i, target_name in enumerate(target_names):
        ax = axes[i]
        
        # Scatter plot of true vs predicted
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=20)
        
        # Error bars (showing uncertainty)
        ax.errorbar(y_true[:, i], y_pred[:, i], yerr=y_std[:, i], 
                   fmt='none', alpha=0.3, capsize=2)
        
        # Perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_xlabel(f'True {target_name}')
        ax.set_ylabel(f'Predicted {target_name}')
        ax.set_title(f'{target_name}: True vs Predicted (with uncertainty)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gp_regressor/gp_regression_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_kernel_info(model: MultiTaskGaussianProcess, target_names: List[str]):
    """
    Print information about the optimized kernels for each target.
    
    Args:
        model: Fitted multi-task Gaussian Process model
        target_names: Names of target variables
    """
    kernels = model.get_kernels()
    log_likelihoods = model.get_log_marginal_likelihood()
    
    print("\n" + "="*60)
    print("OPTIMIZED KERNELS AND MODEL INFORMATION")
    print("="*60)
    
    for target_name in target_names:
        print(f"\n{target_name.upper()}:")
        print("-" * 30)
        print(f"Optimized Kernel: {kernels[target_name]}")
        print(f"Log Marginal Likelihood: {log_likelihoods[target_name]:.4f}")

def main():
    """
    Main function to run the multi-task Gaussian Process analysis.
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
    print("\nTraining multi-task Gaussian Process model...")
    print("This may take a few minutes due to kernel optimization...")
    
    # Try different kernel configurations
    kernels_to_try = [
        ("RBF + White", ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)),
        ("Matern + White", ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0)),
        ("RBF + Matern + White", ConstantKernel(1.0) * (RBF(length_scale=1.0) + Matern(length_scale=1.0, nu=1.5)) + WhiteKernel(noise_level=1.0))
    ]
    
    best_model = None
    best_score = float('inf')
    best_kernel_name = ""
    
    for kernel_name, kernel in kernels_to_try:
        print(f"\nTrying kernel: {kernel_name}")
        try:
            model = MultiTaskGaussianProcess(
                kernel=kernel,
                random_state=42,
                n_restarts_optimizer=5  # Reduced for faster testing
            )
            model.fit(X_train, y_train, feature_names, target_names)
            
            # Quick evaluation
            y_pred = model.predict(X_test)
            score = np.mean([mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])])
            
            if score < best_score:
                best_score = score
                best_model = model
                best_kernel_name = kernel_name
                
            print(f"  Score: {score:.4f}")
            
        except Exception as e:
            print(f"  Error with {kernel_name}: {e}")
            continue
    
    if best_model is None:
        print("All kernels failed. Using default RBF kernel.")
        best_model = MultiTaskGaussianProcess(random_state=42)
        best_model.fit(X_train, y_train, feature_names, target_names)
        best_kernel_name = "Default RBF"
    
    print(f"\nBest kernel: {best_kernel_name}")
    
    # Evaluate model
    print("\nEvaluating model performance...")
    results = evaluate_model(best_model, X_test, y_test, target_names)
    print_evaluation_results(results)
    
    # Print kernel information
    print_kernel_info(best_model, target_names)
    
    # Create visualizations
    print("\nCreating visualizations...")
    y_pred, y_std = best_model.predict(X_test, return_std=True)
    plot_results(y_test, y_pred, y_std, target_names)
    
    # Cross-validation with proper scaling
    print("\nPerforming cross-validation with proper scaling...")
    cv_scores_mse = {}
    cv_scores_mape = {}
    
    for i, target_name in enumerate(target_names):
        # Create pipeline for this target
        pipeline = Pipeline([
            # ('scaler', StandardScaler()),
            ('scaler', MinMaxScaler(feature_range=(0.1, 10.0))),
            ('regressor', GaussianProcessRegressor(
                kernel=best_model.get_kernels()[target_name],
                random_state=42,
                normalize_y=True
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
    results_df.to_csv('gp_regressor/gp_regression_results.csv')
    print(f"\nResults saved to 'gp_regressor/gp_regression_results.csv'")
    
    return best_model, results

if __name__ == "__main__":
    model, results = main()
