"""
Recurrent Neural Network models for time series classification and regression
"""

import torch
import torch.nn as nn
from typing import Literal, Optional


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
        output_size: Optional[int] = None
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
        """
        super().__init__()
        
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.task = task
        
        # Determine output size
        self.output_size = output_size if output_size is not None else num_classes
        
        # Map string name to PyTorch RNN class
        rnn_map = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }
        
        if rnn_type not in rnn_map:
            raise ValueError("rnn_type must be 'RNN', 'LSTM', or 'GRU'")
        
        rnn_module = rnn_map[rnn_type]
        
        # Dropout is only applied between layers (if num_layers > 1)
        dropout_val = dropout_rate if num_layers > 1 else 0
        
        # Create the recurrent layer
        self.rnn = rnn_module(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq_len, features)
            bidirectional=bidirectional,
            dropout=dropout_val
        )
        
        # Calculate input size for the final output layer
        output_input_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Final output layer (works for both classification and regression)
        self.output_layer = nn.Linear(output_input_size, self.output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            output: Output tensor of shape (batch_size, output_size)
                   For classification: logits (raw scores)
                   For regression: predicted values
        """
        # rnn_out shape: (batch_size, seq_len, hidden_size * num_directions)
        rnn_out, hidden = self.rnn(x)
        
        # LSTM returns (h_n, c_n), we only need h_n
        if self.rnn_type == 'LSTM':
            hidden = hidden[0]
        
        # hidden shape: (num_layers * num_directions, batch_size, hidden_size)
        
        if self.bidirectional:
            # Reshape to (num_layers, 2, batch_size, hidden_size)
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
            
            # Concat last fwd (hidden[-1, 0, ...]) and bwd (hidden[-1, 1, ...])
            # Final shape: (batch_size, hidden_size * 2)
            hidden_to_output = torch.cat([hidden[-1, 0, :, :], hidden[-1, 1, :, :]], dim=1)
        else:
            # Take the last layer's hidden state
            # Final shape: (batch_size, hidden_size)
            hidden_to_output = hidden[-1]
        
        # Get output (logits for classification, predictions for regression)
        output = self.output_layer(hidden_to_output)
        return output
