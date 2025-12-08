"""
Advanced normalization strategies
Implements ADVICE 04/12 - Normalisation Strategies
"""
import torch
import torch.nn as nn


class AdaptiveNormalization(nn.Module):
    """
    Adaptive normalization that switches between BatchNorm and GroupNorm
    based on batch size
    
    For small batches (<16), GroupNorm is more stable than BatchNorm.
    For larger batches (>=16), BatchNorm works well.
    """
    def __init__(self, num_channels, num_groups=32, threshold_batch_size=16):
        super().__init__()
        self.threshold = threshold_batch_size
        self.batch_norm = nn.BatchNorm2d(num_channels)
        self.group_norm = nn.GroupNorm(min(num_groups, num_channels), num_channels)
        self.num_channels = num_channels
    
    def forward(self, x):
        batch_size = x.size(0)
        
        if batch_size < self.threshold:
            # Use GroupNorm for small batches
            return self.group_norm(x)
        else:
            # Use BatchNorm for larger batches
            return self.batch_norm(x)


class LayerNorm2d(nn.Module):
    """
    Layer Normalization for 2D feature maps (images)
    Normalizes across channel, height, and width dimensions
    
    Good for: Variable batch sizes, transformers, any single-sample processing
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x):
        # x shape: (B, C, H, W)
        # Normalize over C, H, W for each sample independently
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        var = x.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable affine transform
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        return x


class InstanceNorm2d(nn.Module):
    """
    Instance Normalization for 2D feature maps
    Normalizes across spatial dimensions (H, W) for each channel independently
    
    Good for: Style transfer, image generation, when batch statistics are unreliable
    """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        # x shape: (B, C, H, W)
        # Normalize over H, W for each channel independently
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        return x


def replace_batchnorm_with_groupnorm(model, num_groups=32):
    """
    Replace all BatchNorm layers with GroupNorm in a model
    
    Useful when training with very small batch sizes (common in Colab with limited memory)
    
    Args:
        model: PyTorch model
        num_groups: Number of groups for GroupNorm (default 32)
    
    Returns:
        Modified model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Replace with GroupNorm
            num_channels = module.num_features
            new_module = nn.GroupNorm(min(num_groups, num_channels), num_channels)
            setattr(model, name, new_module)
        else:
            # Recursively apply to submodules
            replace_batchnorm_with_groupnorm(module, num_groups)
    
    return model


def replace_batchnorm_with_layernorm(model):
    """
    Replace all BatchNorm layers with LayerNorm
    
    Useful for: Transformer models, batch-size-independent training
    
    Args:
        model: PyTorch model
    
    Returns:
        Modified model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            new_module = LayerNorm2d(num_channels)
            setattr(model, name, new_module)
        else:
            replace_batchnorm_with_layernorm(module)
    
    return model


def get_normalization_layer(norm_type, num_channels, num_groups=32):
    """
    Factory function to get normalization layer
    
    Args:
        norm_type: 'batch', 'group', 'layer', 'instance', or 'adaptive'
        num_channels: Number of input channels
        num_groups: Number of groups for GroupNorm
    
    Returns:
        Normalization layer
    """
    norm_layers = {
        'batch': lambda: nn.BatchNorm2d(num_channels),
        'group': lambda: nn.GroupNorm(min(num_groups, num_channels), num_channels),
        'layer': lambda: LayerNorm2d(num_channels),
        'instance': lambda: InstanceNorm2d(num_channels),
        'adaptive': lambda: AdaptiveNormalization(num_channels, num_groups)
    }
    
    if norm_type not in norm_layers:
        raise ValueError(f"Normalization type {norm_type} not supported. "
                        f"Choose from: {list(norm_layers.keys())}")
    
    return norm_layers[norm_type]()


def diagnose_batch_size_issues(model, dataloader, device):
    """
    Diagnose if batch size is causing training instability
    
    Compares BatchNorm statistics at different batch sizes
    
    Args:
        model: Model with BatchNorm layers
        dataloader: DataLoader
        device: cuda or cpu
    
    Returns:
        Dictionary with diagnostic information
    """
    model.eval()
    
    bn_layers = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append(module)
    
    if not bn_layers:
        print("⚠️  No BatchNorm layers found in model")
        return {}
    
    print(f"📊 Found {len(bn_layers)} BatchNorm layers")
    print(f"   Current batch size: {dataloader.batch_size}")
    
    # Get one batch and check statistics
    images, _ = next(iter(dataloader))
    images = images.to(device)
    
    with torch.no_grad():
        _ = model(images)
    
    # Check running statistics
    diagnostics = {
        'batch_size': dataloader.batch_size,
        'num_bn_layers': len(bn_layers),
        'running_var_stats': []
    }
    
    for idx, bn in enumerate(bn_layers):
        if bn.running_var is not None:
            var_mean = bn.running_var.mean().item()
            var_std = bn.running_var.std().item()
            diagnostics['running_var_stats'].append({
                'layer': idx,
                'var_mean': var_mean,
                'var_std': var_std,
                'stable': var_std < 1.0  # Heuristic for stability
            })
    
    # Recommendation
    unstable_layers = sum(1 for s in diagnostics['running_var_stats'] if not s['stable'])
    
    if unstable_layers > len(bn_layers) * 0.3:
        print(f"\n⚠️  WARNING: {unstable_layers}/{len(bn_layers)} BatchNorm layers may be unstable")
        print(f"   Consider:")
        print(f"   1. Increasing batch size (current: {dataloader.batch_size})")
        print(f"   2. Using GroupNorm instead of BatchNorm")
        print(f"   3. Using LayerNorm for batch-size independence")
    else:
        print(f"\n✅ BatchNorm statistics look stable")
    
    return diagnostics
