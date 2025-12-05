
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple

class TabularDataset(Dataset):
    """
    Custom PyTorch Dataset for tabular data.
    """
    
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        """
        Args:
            data: Array of shape (n_samples, n_features)
            labels: Array of shape (n_samples,)
        """
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for time series classification.
    """
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Args:
            sequences: Array of shape (n_samples, window_size, n_features)
            labels: Array of shape (n_samples,)
        """
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).long()
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 512,
    num_workers: int = 2,
    shuffle: bool = False,
    drop_last: bool = True,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        X_test: Test sequences
        y_test: Test labels
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle training data
        drop_last: Whether to drop the last incomplete batch
        pin_memory: Whether to use pinned memory (only useful with CUDA)
        
    Returns:
        Training, validation, and test DataLoaders
    """
    if num_workers is None:
        cpu_cores = os.cpu_count() or 2
        num_workers = max(2, min(4, cpu_cores))
    
    # Create TensorDatasets
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    prefetch = 4 if num_workers > 0 else None
    pin_memory_device = "cuda" if (pin_memory and torch.cuda.is_available()) else ""
    
    # Create DataLoaders with performance optimizations
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
        prefetch_factor=prefetch,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle validation
        drop_last=False,  # Don't drop validation samples
        num_workers=num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
        prefetch_factor=prefetch,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle test
        drop_last=False,  # Don't drop test samples
        num_workers=num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
        prefetch_factor=prefetch,
    )
    
    return train_loader, val_loader, test_loader
