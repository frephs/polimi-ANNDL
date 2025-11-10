"""
Model utilities and helper functions
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


def recurrent_summary(model: nn.Module, input_size: Tuple[int, ...]) -> None:
    """
    Custom summary function for recurrent models that correctly
    counts parameters for RNN/GRU/LSTM layers.
    
    Args:
        model: The model to analyze
        input_size: Shape of the input tensor (e.g., (seq_len, features))
    """
    # Dictionary to store output shapes captured by forward hooks
    output_shapes = {}
    # List to track hook handles for later removal
    hooks = []
    
    def get_hook(name):
        """Factory function to create a forward hook for a specific module."""
        def hook(module, input, output):
            # Handle RNN layer outputs (returns a tuple)
            if isinstance(output, tuple):
                # output[0]: all hidden states with shape (batch, seq_len, hidden*directions)
                shape1 = list(output[0].shape)
                shape1[0] = -1  # Replace batch dimension with -1
                
                # output[1]: final hidden state h_n (or tuple (h_n, c_n) for LSTM)
                if isinstance(output[1], tuple):  # LSTM case: (h_n, c_n)
                    shape2 = list(output[1][0].shape)  # Extract h_n only
                else:  # RNN/GRU case: h_n only
                    shape2 = list(output[1].shape)
                
                # Replace batch dimension (middle position) with -1
                shape2[1] = -1
                
                output_shapes[name] = f"[{shape1}, {shape2}]"
            
            # Handle standard layer outputs (e.g., Linear)
            else:
                shape = list(output.shape)
                shape[0] = -1  # Replace batch dimension with -1
                output_shapes[name] = f"{shape}"
        return hook
    
    # 1. Determine the device where model parameters reside
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")  # Fallback for models without parameters
    
    # 2. Create a dummy input tensor with batch_size=1
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # 3. Register forward hooks on target layers
    for name, module in model.named_children():
        if isinstance(module, (nn.Linear, nn.RNN, nn.GRU, nn.LSTM)):
            hook_handle = module.register_forward_hook(get_hook(name))
            hooks.append(hook_handle)
    
    # 4. Execute a dummy forward pass in evaluation mode
    model.eval()
    with torch.no_grad():
        try:
            model(dummy_input)
        except Exception as e:
            print(f"Error during dummy forward pass: {e}")
            for h in hooks:
                h.remove()
            return
    
    # 5. Remove all registered hooks
    for h in hooks:
        h.remove()
    
    # 6. Print the summary table
    print("-" * 79)
    print(f"{'Layer (type)':<25} {'Output Shape':<28} {'Param #':<18}")
    print("=" * 79)
    
    total_params = 0
    total_trainable_params = 0
    
    for name, module in model.named_children():
        if name in output_shapes:
            module_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            total_params += module_params
            total_trainable_params += trainable_params
            
            layer_name = f"{name} ({type(module).__name__})"
            output_shape_str = str(output_shapes[name])
            params_str = f"{trainable_params:,}"
            
            print(f"{layer_name:<25} {output_shape_str:<28} {params_str:<15}")
    
    print("=" * 79)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {total_trainable_params:,}")
    print(f"Non-trainable params: {total_params - total_trainable_params:,}")
    print("-" * 79)


def get_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to place the model on
        
    Returns:
        Initialized model
    """
    from .rnn_models import RecurrentClassifier
    from .feedforward import FeedForwardNet
    
    model_config = config['model']
    model_type = model_config.get('architecture', 'recurrent').lower()
    
    if model_type in ['recurrent', 'rnn', 'lstm', 'gru']:
        model = RecurrentClassifier(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_classes=model_config['num_classes'],
            rnn_type=model_config['type'],
            bidirectional=model_config['bidirectional'],
            dropout_rate=model_config['dropout_rate']
        ).to(device)
    
    elif model_type in ['feedforward', 'ffn', 'mlp']:
        model = FeedForwardNet(
            in_features=model_config['input_size'],
            hidden_layers=model_config.get('hidden_layers', 2),
            hidden_size=model_config['hidden_size'],
            num_classes=model_config['num_classes'],
            dropout_rate=model_config['dropout_rate'],
            activation=model_config.get('activation', 'relu')
        ).to(device)
    
    else:
        raise ValueError(f"Unknown model architecture: {model_type}")
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count the total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
