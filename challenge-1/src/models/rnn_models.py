"""
Recurrent Neural Network models for time series classification and regression
"""

import torch
import torch.nn as nn
from typing import Literal, Optional
import torch.nn.functional as F

class RecurrentNet(nn.Module):
    """
    Generic RNN model supporting RNN, LSTM, and GRU architectures.
    Supports both classification and regression tasks.
    Uses the last hidden state for prediction.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        rnn_type: Literal['RNN', 'LSTM', 'GRU'] = 'GRU',
        bidirectional: bool = False,
        dropout_rate: float = 0.2,
        task: Literal['classification', 'regression'] = 'classification',
        output_size: Optional[int] = None,
        use_conv1d: bool = False,
        conv1d_filters: list = None,
        conv1d_kernel_sizes: list = None,
        conv1d_dropout: float = 0.1
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of features in the hidden state
            num_layers: Number of recurrent layers
            num_classes: Number of output classes (for classification) or output dimensions
            rnn_type: Type of RNN ('RNN', 'LSTM', or 'GRU')
            bidirectional: If True, becomes a bidirectional RNN
            dropout_rate: Dropout probability (applied between layers if num_layers > 1)
            task: Type of task ('classification' or 'regression')
            output_size: Number of output dimensions (for regression). If None, uses num_classes
            use_conv1d: If True, adds 1D convolutional layers before RNN
            conv1d_filters: List of output channels for each Conv1D layer (e.g., [64, 128])
            conv1d_kernel_sizes: List of kernel sizes for each Conv1D layer (e.g., [3, 3])
            conv1d_dropout: Dropout rate for Conv1D layers
        """
        super().__init__()
        
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.task = task
        self.use_conv1d = use_conv1d
        
        # Determine output size
        self.output_size = output_size if output_size is not None else num_classes
        
        if self.use_conv1d and conv1d_filters and conv1d_kernel_sizes:
            self.conv_layers = nn.ModuleList()
            if conv1d_filters is None:
                conv1d_filters = [64]
            if conv1d_kernel_sizes is None:
                conv1d_kernel_sizes = [3] * len(conv1d_filters)
            in_channels = input_size
            for i, (out_channels, kernel_size) in enumerate(zip(conv1d_filters, conv1d_kernel_sizes)):
                conv_block = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(conv1d_dropout)
                )
                self.conv_layers.append(conv_block)
                in_channels = out_channels
            rnn_input_size = conv1d_filters[-1]
        else:
            self.conv_layers = None
            rnn_input_size = input_size
        
        rnn_map = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}
        if rnn_type not in rnn_map:
            raise ValueError("rnn_type must be 'RNN', 'LSTM', or 'GRU'")
        rnn_module = rnn_map[rnn_type]
        dropout_val = dropout_rate if num_layers > 1 else 0
        
        # Create the recurrent layer
        self.rnn = rnn_module(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_val
        )
        
        # Calculate input size for the final output layer
        output_input_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention Layer 
        self.attention_scorer = nn.Linear(output_input_size, 1)
        self.output_layer = nn.Linear(output_input_size, self.output_size)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1D Layers
        if self.use_conv1d and self.conv_layers is not None:
            x = x.transpose(1, 2)
            for conv_block in self.conv_layers:
                x = conv_block(x)
            x = x.transpose(1, 2)
        
        # RNN Layer
        rnn_out, hidden = self.rnn(x)
        
        # Attention Mechanism
        energy = torch.tanh(self.attention_scorer(rnn_out))
        attention_weights = F.softmax(energy, dim=1)
        context_vector = rnn_out * attention_weights
        context_vector = torch.sum(context_vector, dim=1)
        
        # Feed the final context vector into the classifier
        output = self.output_layer(context_vector)
        return output
