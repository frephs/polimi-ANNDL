"""
Configuration schema validation for Pirates Pain Classification.
Ensures config files are valid and complete before training.
"""

from typing import Dict, Any, List, Union
import warnings


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


# Define the configuration schema
CONFIG_SCHEMA = {
    'seed': {
        'type': int,
        'required': True,
        'min': 0,
        'description': 'Random seed for reproducibility'
    },
    'data': {
        'type': dict,
        'required': True,
        'schema': {
            'raw_dir': {'type': str, 'required': True},
            'processed_dir': {'type': str, 'required': True},
            'train_file': {'type': str, 'required': True},
            'train_labels_file': {'type': str, 'required': True},
            'test_file': {'type': str, 'required': True},
        }
    },
    'preprocessing': {
        'type': dict,
        'required': True,
        'schema': {
            'enabled': {'type': bool, 'required': True},
            #'scale_features': {'type': bool, 'required': True},
            #'skew_threshold': {'type': (int, float), 'required': True, 'min': 0},
            #'clip_quantiles': {'type': list, 'required': True, 'length': 2},
            'drop_features': {'type': list, 'required': True},
            'combine_correlations': {'type': bool, 'required': True},
            'create_prosthesis_flag': {'type': bool, 'required': True},
        }
    },
    'time_features': {
        'type': dict,
        'required': False,
        'schema': {
            'enabled': {'type': bool, 'required': True},
            'extract_hour': {'type': bool, 'required': False},
            'extract_day_of_week': {'type': bool, 'required': False},
            'extract_day_of_month': {'type': bool, 'required': False},
            'use_cyclical_encoding': {'type': bool, 'required': False},
        }
    },
    'labels': {
        'type': dict,
        'required': True,
        'schema': {
            'no_pain': {'type': int, 'required': True},
            'low_pain': {'type': int, 'required': True},
            'high_pain': {'type': int, 'required': True},
        }
    },
    'split': {
        'type': dict,
        'required': True,
        'schema': {
            'val_size': {'type': (int, float), 'required': True, 'min': 0.0, 'max': 1.0},
            'stratify': {'type': bool, 'required': True},
        }
    },
    'sequences': {
        'type': dict,
        'required': True,
        'schema': {
            'window_size': {'type': int, 'required': True, 'min': 1, 'max': 160},
            'stride': {'type': int, 'required': True, 'min': 1},
        }
    },
    'augmentation': {
        'type': dict,
        'required': True,
        'schema': {
            'enabled': {'type': bool, 'required': True},
            'oversample': {'type': bool, 'required': True},
            'target_distribution': {'type': str, 'required': True, 'choices': ['balanced', 'majority']},
            'augment_duplicates': {'type': bool, 'required': True},
            'noise_level': {'type': (int, float), 'required': True, 'min': 0},
            'scale_range': {'type': list, 'required': True, 'length': 2},
            'shift_range': {'type': int, 'required': True, 'min': 0},
            'apply_noise': {'type': bool, 'required': True},
            'apply_scaling': {'type': bool, 'required': True},
            'apply_shift': {'type': bool, 'required': True},
        }
    },
    'model': {
        'type': dict,
        'required': True,
        'schema': {
            'type': {'type': str, 'required': True},
            'input_size': {'type': (int, type(None)), 'required': False},
            'hidden_size': {'type': int, 'required': True, 'min': 1},
            'num_layers': {'type': int, 'required': True, 'min': 1},
            'num_classes': {'type': int, 'required': True, 'min': 2},
            'rnn_type': {'type': str, 'required': True, 'choices': ['RNN', 'LSTM', 'GRU']},
            'bidirectional': {'type': bool, 'required': True},
            'dropout_rate': {'type': (int, float), 'required': True, 'min': 0.0, 'max': 1.0},
            'task': {'type': str, 'required': True, 'choices': ['classification', 'regression']},
            'use_conv1d': {'type': bool, 'required': False},
            'conv1d_filters': {'type': list, 'required': False},
            'conv1d_kernel_sizes': {'type': list, 'required': False},
            'conv1d_dropout': {'type': (int, float), 'required': False, 'min': 0.0, 'max': 1.0},
        }
    },
    'training': {
        'type': dict,
        'required': True,
        'schema': {
            'epochs': {'type': int, 'required': True, 'min': 1},
            'batch_size': {'type': int, 'required': True, 'min': 1},
            'learning_rate': {'type': (int, float), 'required': True, 'min': 0},
            'l1_lambda': {'type': (int, float), 'required': True, 'min': 0},
            'l2_lambda': {'type': (int, float), 'required': True, 'min': 0},
            #'dropout_rate': {'type': (int, float), 'required': True, 'min': 0.0, 'max': 1.0},
            'patience': {'type': int, 'required': True, 'min': 1},
            'evaluation_metric': {'type': str, 'required': True},
            'mode': {'type': str, 'required': True, 'choices': ['max', 'min']},
            'restore_best_weights': {'type': bool, 'required': True},
            'scheduler': {
                'type': dict,
                'required': True,
                'schema': {
                    'enabled': {'type': bool, 'required': True},
                    'type': {'type': str, 'required': False},
                    'factor': {'type': (int, float), 'required': False, 'min': 0, 'max': 1},
                    'patience': {'type': int, 'required': False, 'min': 1},
                    'min_lr': {'type': (int, float), 'required': False, 'min': 0},
                }
            },
            'verbose': {'type': int, 'required': True, 'min': 1},
        }
    },
    'dataloader': {
        'type': dict,
        'required': True,
        'schema': {
            'num_workers': {'type': int, 'required': True, 'min': 0},
            'shuffle_train': {'type': bool, 'required': True},
            'shuffle_val': {'type': bool, 'required': True},
            'drop_last': {'type': bool, 'required': True},
            'pin_memory': {'type': bool, 'required': True},
            'prefetch_factor': {'type': int, 'required': True, 'min': 2},
        }
    },
    'logging': {
        'type': dict,
        'required': True,
        'schema': {
            'tensorboard_dir': {'type': str, 'required': True},
            'save_dir': {'type': str, 'required': True},
            'experiment_name': {'type': str, 'required': True},
        }
    },
    'cross_validation': {
        'type': dict,
        'required': True,
        'schema': {
            'enabled': {'type': bool, 'required': True},
            'k_folds': {'type': int, 'required': True, 'min': 2},
        }
    },
    'hyperparameter_tuning': {
        'type': dict,
        'required': True,
        'schema': {
            'enabled': {'type': bool, 'required': True},
            'param_grid': {'type': dict, 'required': True},
        }
    },
}


def validate_value(value: Any, spec: Dict[str, Any], path: str = "") -> List[str]:
    """
    Validate a single value against its specification.
    
    Args:
        value: The value to validate
        spec: The specification dictionary
        path: The current path in the config (for error messages)
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check type
    expected_type = spec['type']
    if isinstance(expected_type, tuple):
        if not isinstance(value, expected_type):
            errors.append(f"{path}: Expected one of {expected_type}, got {type(value).__name__}")
    else:
        if not isinstance(value, expected_type):
            errors.append(f"{path}: Expected {expected_type.__name__}, got {type(value).__name__}")
    
    # Check min/max for numeric values
    if isinstance(value, (int, float)):
        if 'min' in spec and value < spec['min']:
            errors.append(f"{path}: Value {value} is less than minimum {spec['min']}")
        if 'max' in spec and value > spec['max']:
            errors.append(f"{path}: Value {value} is greater than maximum {spec['max']}")
    
    # Check choices for string values
    if isinstance(value, str) and 'choices' in spec:
        if value not in spec['choices']:
            errors.append(f"{path}: Value '{value}' not in allowed choices {spec['choices']}")
    
    # Check length for lists
    if isinstance(value, list) and 'length' in spec:
        if len(value) != spec['length']:
            errors.append(f"{path}: Expected list of length {spec['length']}, got {len(value)}")
    
    # Validate nested dictionary
    if isinstance(value, dict) and 'schema' in spec:
        errors.extend(validate_config_recursive(value, spec['schema'], path))
    
    return errors


def validate_config_recursive(config: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> List[str]:
    """
    Recursively validate a configuration dictionary against a schema.
    
    Args:
        config: The configuration dictionary to validate
        schema: The schema dictionary
        path: The current path in the config (for error messages)
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check for required fields
    for key, spec in schema.items():
        full_path = f"{path}.{key}" if path else key
        
        if spec.get('required', False) and key not in config:
            errors.append(f"Missing required field: {full_path}")
            continue
        
        if key in config:
            errors.extend(validate_value(config[key], spec, full_path))
    
    # Check for unexpected fields (warnings only)
    for key in config:
        if key not in schema:
            warnings.warn(f"Unexpected field in config: {path}.{key}" if path else f"Unexpected field in config: {key}")
    
    return errors


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate a configuration dictionary against the schema.
    
    Args:
        config: The configuration dictionary to validate
        
    Raises:
        ConfigValidationError: If the configuration is invalid
    """
    errors = validate_config_recursive(config, CONFIG_SCHEMA)
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigValidationError(error_msg)
    
    print("✓ Configuration validation passed")


def get_config_template() -> Dict[str, Any]:
    """
    Get a template configuration with default values.
    
    Returns:
        Template configuration dictionary
    """
    return {
        'seed': 42,
        'data': {
            'raw_dir': '../data/raw',
            'processed_dir': '../data/processed',
            'train_file': 'pirate_pain_train.csv',
            'train_labels_file': 'pirate_pain_train_labels.csv',
            'test_file': 'pirate_pain_test.csv',
        },
        'preprocessing': {
            'enabled': True,
            'scale_features': True,
            'skew_threshold': 5.0,
            'clip_quantiles': [0.01, 0.99],
            'drop_features': ['joint_30'],
            'combine_correlations': True,
            'create_prosthesis_flag': True,
        },
        'labels': {
            'no_pain': 0,
            'low_pain': 1,
            'high_pain': 2,
        },
        'split': {
            'val_size': 0.15,
            'stratify': True,
        },
        'sequences': {
            'window_size': 160,
            'stride': 160,
        },
        'augmentation': {
            'enabled': False,
            'oversample': False,
            'target_distribution': 'balanced',
            'augment_duplicates': True,
            'noise_level': 0.01,
            'scale_range': [0.95, 1.05],
            'shift_range': 5,
            'apply_noise': True,
            'apply_scaling': True,
            'apply_shift': True,
        },
        'model': {
            'type': 'RecurrentClassifier',
            'input_size': None,
            'hidden_size': 128,
            'num_layers': 2,
            'num_classes': 3,
            'rnn_type': 'LSTM',
            'bidirectional': False,
            'dropout_rate': 0.2,
            'task': 'classification',
        },
        'training': {
            'epochs': 500,
            'batch_size': 64,
            'learning_rate': 0.001,
            'l1_lambda': 0.0,
            'l2_lambda': 0.001,
            'dropout_rate': 0.3,
            'patience': 50,
            'evaluation_metric': 'val_f1',
            'mode': 'max',
            'restore_best_weights': True,
            'scheduler': {
                'enabled': True,
                'type': 'ReduceLROnPlateau',
                'factor': 0.5,
                'patience': 10,
                'min_lr': 1.0e-6,
            },
            'verbose': 10,
        },
        'dataloader': {
            'num_workers': 2,
            'shuffle_train': True,
            'shuffle_val': False,
            'drop_last': False,
            'pin_memory': True,
            'prefetch_factor': 4,
        },
        'logging': {
            'tensorboard_dir': './tensorboard',
            'save_dir': './models',
            'experiment_name': 'pirates_pain_classification',
        },
        'cross_validation': {
            'enabled': False,
            'k_folds': 5,
        },
        'hyperparameter_tuning': {
            'enabled': False,
            'param_grid': {
                'hidden_size': [64, 128],
                'num_layers': [1, 2],
                'learning_rate': [0.001, 0.01],
                'dropout_rate': [0.1, 0.2],
            },
        },
    }
