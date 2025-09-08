"""
Dynamic data utilities wrapper that imports the appropriate data utility based on dataset.
"""
import sys
import os

def get_data_utils(dataset_name):
    """
    Dynamically import the appropriate data utility based on dataset name.
    
    Args:
        dataset_name: 'ENB', 'SYN', or 'POS'
    
    Returns:
        Module with data utility functions
    """
    if dataset_name.upper() == 'ENB':
        from data_utils_ENB import (
            load_and_preprocess_data,
            get_train_test_split,
            create_standard_scaler,
            get_data_info,
            validate_data_consistency
        )
    elif dataset_name.upper() == 'SYN':
        from data_utils_SYN import (
            load_and_preprocess_data,
            get_train_test_split,
            create_standard_scaler,
            get_data_info,
            validate_data_consistency
        )
    elif dataset_name.upper() == 'POS':
        from data_utils_POS import (
            load_and_preprocess_data,
            get_train_test_split,
            create_standard_scaler,
            get_data_info,
            validate_data_consistency
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be 'ENB', 'SYN', or 'POS'")
    
    return {
        'load_and_preprocess_data': load_and_preprocess_data,
        'get_train_test_split': get_train_test_split,
        'create_standard_scaler': create_standard_scaler,
        'get_data_info': get_data_info,
        'validate_data_consistency': validate_data_consistency
    }
