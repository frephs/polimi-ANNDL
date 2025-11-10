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
