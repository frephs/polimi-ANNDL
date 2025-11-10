"""
Feedforward Neural Network models for classification and regression tasks
"""

import torch
import torch.nn as nn
from typing import Optional, Literal


class FeedForwardNet(nn.Module):
    """
    Simple feedforward neural network with configurable architecture.
    Supports both classification and regression tasks.
    Supports dropout regularization for preventing overfitting.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_layers: int = 1,
        hidden_size: int = 128,
        num_classes: int = 10,
        dropout_rate: float = 0.0,
        activation: str = 'relu',
        task: Literal['classification', 'regression'] = 'classification',
        output_size: Optional[int] = None
    ):
        """
        Args:
            in_features: Number of input features
            hidden_layers: Number of hidden layers (not counting input/output)
            hidden_size: Number of neurons in each hidden layer
            num_classes: Number of output classes (for classification) or output dimensions
            dropout_rate: Dropout probability (0 means no dropout)
            activation: Activation function ('relu', 'tanh', 'gelu', 'leaky_relu')
            task: Type of task ('classification' or 'regression')
            output_size: Number of output dimensions (for regression). If None, uses num_classes
        """
        super().__init__()
        
        self.in_features = in_features
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.task = task
        
        # Determine output size
        self.output_size = output_size if output_size is not None else num_classes
        
        # Map activation names to PyTorch modules
        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'gelu': nn.GELU,
            'leaky_relu': nn.LeakyReLU
        }
        
        if activation.lower() not in activation_map:
            raise ValueError(f"activation must be one of {list(activation_map.keys())}")
        
        activation_fn = activation_map[activation.lower()]
        
        # Build the network layers
        modules = []
        
        # First layer: input -> hidden
        modules.append(nn.Linear(in_features, hidden_size))
        if dropout_rate > 0:
            modules.append(nn.Dropout(dropout_rate))
        modules.append(activation_fn())
        
        # Additional hidden layers
        for _ in range(hidden_layers):
            modules.append(nn.Linear(hidden_size, hidden_size))
            if dropout_rate > 0:
                modules.append(nn.Dropout(dropout_rate))
            modules.append(activation_fn())
        
        # Output layer (works for both classification and regression)
        modules.append(nn.Linear(hidden_size, self.output_size))
        
        self.net = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            output: Output tensor of shape (batch_size, output_size)
                   For classification: logits (raw scores)
                   For regression: predicted values
        """
        return self.net(x)
    
    def __repr__(self) -> str:
        return (f"FeedForwardNet(in_features={self.in_features}, "
                f"hidden_layers={self.hidden_layers}, "
                f"hidden_size={self.hidden_size}, "
                f"num_classes={self.num_classes}, "
                f"dropout_rate={self.dropout_rate})")


class ResidualBlock(nn.Module):
    """
    Residual block for deeper feedforward networks.
    Uses skip connections to help gradient flow.
    """
    
    def __init__(self, hidden_size: int, dropout_rate: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.layer_norm(out + residual)
        return self.relu(out)


class ResidualFeedForwardNet(nn.Module):
    """
    Feedforward network with residual connections for better gradient flow.
    Supports both classification and regression tasks.
    Suitable for deeper architectures.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_size: int = 128,
        num_blocks: int = 3,
        num_classes: int = 10,
        dropout_rate: float = 0.1,
        task: Literal['classification', 'regression'] = 'classification',
        output_size: Optional[int] = None
    ):
        """
        Args:
            in_features: Number of input features
            hidden_size: Size of hidden layers
            num_blocks: Number of residual blocks
            num_classes: Number of output classes (for classification) or output dimensions
            dropout_rate: Dropout probability
            task: Type of task ('classification' or 'regression')
            output_size: Number of output dimensions (for regression). If None, uses num_classes
        """
        super().__init__()
        
        self.task = task
        self.output_size = output_size if output_size is not None else num_classes
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate) 
            for _ in range(num_blocks)
        ])
        
        # Output layer (works for both classification and regression)
        self.output = nn.Linear(hidden_size, self.output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)
