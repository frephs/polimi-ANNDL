"""
Device utilities for PyTorch.
"""

import torch


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device_str: Device string ('cuda', 'cpu', or 'auto')
        
    Returns:
        PyTorch device
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    return device
