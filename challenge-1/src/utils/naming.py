"""
Utilities for generating descriptive experiment names.
"""

from datetime import datetime
from typing import Dict, Any


def generate_experiment_name(config: Dict[str, Any], f1_score: float = None) -> str:
    """
    Generate a descriptive experiment name from configuration.
    
    Format: {rnn_type}_{bidir}_{hidden}h_{layers}l_{lr}lr_{timestamp}[_{f1}f1]
    
    Args:
        config: Configuration dictionary
        f1_score: Optional F1 score to append
        
    Returns:
        Descriptive experiment name string
    """
    parts = []
    
    # Model architecture
    model_config = config.get('model', {})
    
    # RNN type (LSTM, GRU, RNN)
    rnn_type = model_config.get('rnn_type', 'RNN')
    parts.append(rnn_type)
    
    # Bidirectional
    bidir = 'bi' if model_config.get('bidirectional', False) else 'uni'
    parts.append(bidir)
    
    # Hidden size
    hidden_size = model_config.get('hidden_size', 128)
    parts.append(f"{hidden_size}h")
    
    # Number of layers
    num_layers = model_config.get('num_layers', 2)
    parts.append(f"{num_layers}l")
    
    # Learning rate (compact format)
    training_config = config.get('training', {})
    lr = training_config.get('learning_rate', 0.001)
    lr_str = f"{lr:.0e}".replace('e-0', 'e-').replace('e-', 'e')
    if lr >= 0.0001:
        # Use decimal notation for common values
        lr_str = f"{lr:.4f}".rstrip('0').rstrip('.')
    parts.append(f"{lr_str}lr")
    
    # Dropout (if non-zero and not default)
    dropout = model_config.get('dropout_rate', 0.0)
    if dropout > 0 and dropout != 0.3:
        parts.append(f"d{dropout:.2f}".replace('.', ''))
    
    # Weight decay (if non-zero)
    weight_decay = training_config.get('weight_decay', 0.0)
    if weight_decay > 0:
        wd_str = f"{weight_decay:.0e}".replace('e-0', 'e-').replace('e-', 'e')
        parts.append(f"wd{wd_str}")
    
    # Batch size (if not default 64)
    batch_size = training_config.get('batch_size', 64)
    if batch_size != 64:
        parts.append(f"bs{batch_size}")
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base name
    experiment_name = "_".join(parts) + f"_{timestamp}"
    
    # Add F1 score if provided
    if f1_score is not None:
        f1_str = f"{f1_score:.4f}".replace('.', '')
        experiment_name += f"_f1{f1_str}"
    
    return experiment_name


def generate_model_filename(
    config: Dict[str, Any],
    epoch: int,
    metric_value: float,
    metric_name: str = 'f1',
    timestamp: str = None
) -> str:
    """
    Generate a descriptive model filename.
    
    Format: {rnn_type}_{bidir}_{hidden}h_{layers}l_{timestamp}_ep{epoch}_best_{metric}{value}.pt
    
    Args:
        config: Configuration dictionary
        epoch: Training epoch number
        metric_value: Best metric value
        metric_name: Name of metric
        timestamp: Optional timestamp string
        
    Returns:
        Model filename string
    """
    parts = []
    
    # Model architecture
    model_config = config.get('model', {})
    
    rnn_type = model_config.get('rnn_type', 'RNN')
    parts.append(rnn_type)
    
    bidir = 'bi' if model_config.get('bidirectional', False) else 'uni'
    parts.append(bidir)
    
    hidden_size = model_config.get('hidden_size', 128)
    parts.append(f"{hidden_size}h")
    
    num_layers = model_config.get('num_layers', 2)
    parts.append(f"{num_layers}l")
    
    # Timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp)
    
    # Epoch
    parts.append(f"ep{epoch:03d}")
    
    # Metric
    metric_str = f"{metric_name}{metric_value:.4f}".replace('.', '')
    parts.append(f"best_{metric_str}")
    
    return "_".join(parts) + ".pt"


def parse_experiment_name(name: str) -> Dict[str, Any]:
    """
    Parse experiment name back into components.
    
    Args:
        name: Experiment name string
        
    Returns:
        Dictionary with parsed components
    """
    components = {}
    parts = name.split('_')
    
    try:
        # RNN type
        components['rnn_type'] = parts[0]
        
        # Bidirectional
        components['bidirectional'] = parts[1] == 'bi'
        
        # Parse remaining parts
        for part in parts[2:]:
            if part.endswith('h'):
                components['hidden_size'] = int(part[:-1])
            elif part.endswith('l'):
                components['num_layers'] = int(part[:-1])
            elif part.endswith('lr'):
                lr_str = part[:-2]
                components['learning_rate'] = float(lr_str)
            elif part.startswith('d') and len(part) <= 4:
                components['dropout'] = float(part[1:]) / 100
            elif part.startswith('wd'):
                components['weight_decay'] = float(part[2:])
            elif part.startswith('bs'):
                components['batch_size'] = int(part[2:])
            elif part.startswith('f1'):
                f1_str = part[2:]
                components['f1_score'] = float(f1_str) / 10000
            elif len(part) == 8 and part.isdigit():
                components['date'] = part
            elif len(part) == 6 and part.isdigit():
                components['time'] = part
                components['timestamp'] = f"{components.get('date', '')}_{part}"
    
    except (ValueError, IndexError):
        pass
    
    return components
