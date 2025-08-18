import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Global constants for consistent data handling
RANDOM_STATE = 42
TEST_SIZE = 0.2
SCALER_FEATURE_RANGE = (0.1, 10.0)
cDATA_FILE_PATH = '../data/syn_reg.csv'

def load_and_preprocess_data(file_path: str = DATA_FILE_PATH) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load and preprocess the ENB2012 dataset with standardized approach.
    
    Args:
        file_path: Path to the CSV file (defaults to standard path)
        
    Returns:
        X: Features array
        y: Targets array
        feature_names: List of feature names
        target_names: List of target names
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Separate features and targets
    feature_cols = [col for col in df.columns if col.startswith('f_')]
    target_cols = [col for col in df.columns if col.startswith('target')]
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    feature_names = feature_cols
    target_names = target_cols
    
    return X, y, feature_names, target_names

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

def create_standard_scaler(feature_range: Tuple[float, float] = SCALER_FEATURE_RANGE) -> MinMaxScaler:
    """
    Create a standardized MinMaxScaler with consistent parameters.
    
    Args:
        feature_range: Range for scaling (default: (0.1, 10.0))
        
    Returns:
        Configured MinMaxScaler
    """
    return MinMaxScaler(feature_range=feature_range)

def get_data_info(X: np.ndarray, y: np.ndarray, feature_names: List[str], target_names: List[str]) -> None:
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
    
    # Print scaling information
    print(f"Scaling: MinMaxScaler {SCALER_FEATURE_RANGE}")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Test size: {TEST_SIZE}")

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
    if not all(name.startswith('X') or name.startswith('f_') for name in feature_names):
        raise ValueError("All feature names should start with 'X' or 'f_'")
    
    if not all(name.startswith('target') for name in target_names):
        raise ValueError("All target names should start with 'target'")
    
    return True 