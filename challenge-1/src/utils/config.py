"""
Configuration utilities for loading and managing experiment configurations.
"""

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str, validate: bool = True) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        validate: Whether to validate the config against the schema
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigValidationError: If config validation fails
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if validate:
        from .config_schema import validate_config
        validate_config(config)
    
    return config


def update_config(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        config: Base configuration dictionary
        **kwargs: Key-value pairs to update (supports nested keys with dot notation)
        
    Returns:
        Updated configuration dictionary
    """
    import copy
    updated_config = copy.deepcopy(config)
    
    for key, value in kwargs.items():
        keys = key.split('.')
        d = updated_config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    
    return updated_config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """
    Pretty print configuration dictionary.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print(' ' * indent + f"{key}:")
            print_config(value, indent + 2)
        else:
            print(' ' * indent + f"{key}: {value}")
