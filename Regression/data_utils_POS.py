import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Global constants for consistent data handling
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_POSITIVE = 1e-2  # Minimum positive value for positivity clamp
DATA_FILE_PATH = '../data/pos_reg.csv'

def load_and_preprocess_data(file_path: str = DATA_FILE_PATH, apply_scaling: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load and preprocess the dataset with MinMaxScaler to range (0.1, 10) for positive values.
    
    Args:
        file_path: Path to the CSV file (defaults to standard path)
        apply_scaling: Whether to apply MinMaxScaler (default: True)
        
    Returns:
        X: Features array
        y: Targets array
        feature_names: List of feature names
        target_names: List of target names
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Separate features and targets
    # For POS dataset: features are X1, X2, X3... and targets are Y1, Y2
    feature_cols = [col for col in df.columns if col.startswith('f_')]
    target_cols = [col for col in df.columns if col.startswith('target')]
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # Apply MinMaxScaler to range (0.1, 10) for positive values (if requested)
    if apply_scaling:
        X = minmax_scale_positive(X)
        y = minmax_scale_positive(y)
    
    feature_names = feature_cols
    target_names = target_cols
    
    return X, y, feature_names, target_names

def minmax_scale_positive(X, feature_range=(0.1, 10)):
    """
    MinMaxScaler to range (0.1, 10) for positive values only.
    Works for both features (X) and targets (Y).
    """
    print(f"\n🔧 MINMAX SCALING DETAILS:")
    print(f"   Input shape: {X.shape}")
    print(f"   Feature range: {feature_range}")
    print(f"   Min positive value: {MIN_POSITIVE}")
    
    arr = np.asarray(X, dtype=float).copy()
    
    # Handle any non-finite values
    arr = np.where(np.isfinite(arr), arr, 0.0)
    
    # Apply MinMaxScaler to range (0.1, 10)
    scaler = MinMaxScaler(feature_range=feature_range)
    arr_scaled = scaler.fit_transform(arr)
    
    # Ensure all values are positive (should already be due to range)
    arr_scaled = np.maximum(arr_scaled, feature_range[0])
    
    # Print scaling stats
    print(f"   Min value: {arr_scaled.min():.6f}")
    print(f"   Max value: {arr_scaled.max():.6f}")
    print(f"   Mean value: {arr_scaled.mean():.6f}")
    print(f"   Std value: {arr_scaled.std():.6f}")
    print(f"   Values < min_range: {np.sum(arr_scaled < feature_range[0])}")
    
    return arr_scaled

def get_train_test_split(X: np.ndarray, y: np.ndarray, 
                        test_size: float = TEST_SIZE, 
                        random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get standardized train/test split.
    
    Args:
        X: Features array
        y: Targets array
        test_size: Proportion of test set (default: 0.2)
        random_state: Random seed (default: 42)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_standard_scaler(feature_range: Tuple[float, float] = None) -> 'MinMaxProcessor':
    """
    Create a standardized MinMaxProcessor with consistent parameters.
    
    Args:
        feature_range: Range for MinMaxScaler (default: (0.1, 10))
        
    Returns:
        Configured MinMaxProcessor
    """
    if feature_range is None:
        feature_range = (0.1, 10)
    return NoOpScaler()

class NoOpScaler:
    """
    No-operation scaler that returns data unchanged.
    Used when data is already scaled in load_and_preprocess_data().
    """
    def fit_transform(self, X, y=None):
        """Return data unchanged."""
        return X
    
    def transform(self, X):
        """Return data unchanged."""
        return X
    
    def fit(self, X, y=None):
        """No operation needed."""
        return self

class MinMaxProcessor:
    """
    Wrapper class to maintain compatibility with existing code while using MinMaxScaler.
    """
    def __init__(self, feature_range=(0.1, 10)):
        self.feature_range = feature_range
        self.scaler = MinMaxScaler(feature_range=feature_range)
    
    def fit_transform(self, X, y=None):
        """Apply MinMaxScaler to training data."""
        return minmax_scale_positive(X, self.feature_range)
    
    def transform(self, X):
        """Apply MinMaxScaler to test data."""
        return minmax_scale_positive(X, self.feature_range)
    
    def fit(self, X, y=None):
        """Fit the scaler."""
        self.scaler.fit(X)
        return self

def get_data_info(X: np.ndarray, y: np.ndarray, feature_names: List[str], target_names: List[str]) -> Dict:
    """
    Print standardized data information.
    
    Args:
        X: Features array
        y: Targets array
        feature_names: List of feature names
        target_names: List of target names
    """
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Targets: {target_names}")
    
    # Split data to get training/test sizes
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Print preprocessing information
    print(f"Preprocessing: MinMaxScaler to range (0.1, 10) + positivity clamp")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Test size: {TEST_SIZE}")
    
    return {
        'num_features': X.shape[1],
        'num_targets': y.shape[1],
        'num_samples': X.shape[0],
        'feature_names': feature_names,
        'target_names': target_names,
        'feature_range': (X.min(), X.max()),
        'target_range': (y.min(), y.max())
    }

def validate_data_consistency(X: np.ndarray, y: np.ndarray, feature_names: List[str], target_names: List[str]) -> bool:
    """
    Validate that data follows expected format and dimensions.
    
    Args:
        X: Features array
        y: Targets array
        feature_names: List of feature names
        target_names: List of target names
        
    Returns:
        True if data is consistent, raises ValueError otherwise
    """
    if X.shape[1] != len(feature_names):
        raise ValueError(f"Feature dimensions mismatch: X has {X.shape[1]} features but {len(feature_names)} feature names")
    
    if y.shape[1] != len(target_names):
        raise ValueError(f"Target dimensions mismatch: y has {y.shape[1]} targets but {len(target_names)} target names")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Sample count mismatch: X has {X.shape[0]} samples but y has {y.shape[0]} samples")
    
    # Check for expected feature/target naming convention
    if not all(name.startswith('f_') for name in feature_names):
        raise ValueError("All feature names should start with 'f_' (POS format)")
    
    if not all(name.startswith('target') for name in target_names):
        raise ValueError("All target names should start with 'target' (POS format)")
    
    return True
