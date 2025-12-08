"""
Visualization utilities for activation maps and attention mechanisms
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2


class ActivationMapExtractor:
    """Extract and visualize activation maps from model layers"""
    
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: The neural network model
            target_layer: Specific layer to hook (if None, uses last conv layer)
        """
        self.model = model
        self.activations = []
        self.gradients = []
        self.hooks = []
        
        # Auto-detect last convolutional layer if not specified
        if target_layer is None:
            target_layer = self._find_last_conv_layer()
        
        self.target_layer = target_layer
        
        # Register hooks
        if target_layer is not None:
            self._register_hooks(target_layer)
    
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model"""
        last_conv = None
        
        # Handle AttentionWrapper
        model_to_search = self.model.backbone if hasattr(self.model, 'backbone') else self.model
        
        for name, module in model_to_search.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        
        return last_conv
    
    def _register_hooks(self, layer):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())
        
        self.hooks.append(layer.register_forward_hook(forward_hook))
        self.hooks.append(layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activation_map(self, image, target_class=None):
        """
        Generate activation map for an image
        
        Args:
            image: Input tensor (1, C, H, W) or (C, H, W)
            target_class: Target class for Grad-CAM (if None, uses predicted class)
        
        Returns:
            activation_map: 2D numpy array
            predicted_class: Predicted class index
        """
        self.model.eval()
        self.activations = []
        self.gradients = []
        
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(next(self.model.parameters()).device)
        image.requires_grad = True
        
        # Forward pass
        output = self.model(image)
        predicted_class = output.argmax(dim=1).item()
        
        # Use predicted class if not specified
        if target_class is None:
            target_class = predicted_class
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get activation and gradient
        activation = self.activations[0]  # (1, C, H, W)
        gradient = self.gradients[0]  # (1, C, H, W)
        
        # Grad-CAM: weight channels by gradients
        weights = gradient.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activation).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, predicted_class
    
    def visualize_activation(self, image, original_image=None, target_class=None, 
                           alpha=0.4, cmap='jet', title=None):
        """
        Visualize activation map overlaid on original image
        
        Args:
            image: Input tensor (1, C, H, W) or (C, H, W) - preprocessed
            original_image: Original image tensor or numpy array (H, W, C) for overlay
            target_class: Target class for visualization
            alpha: Overlay transparency
            cmap: Colormap for heatmap
            title: Plot title
        
        Returns:
            fig: Matplotlib figure
        """
        # Get activation map
        cam, pred_class = self.get_activation_map(image, target_class)
        
        # Prepare original image for overlay
        if original_image is None:
            # Use preprocessed image (denormalize)
            img_display = image.squeeze().cpu().permute(1, 2, 0).numpy()
            # Simple denormalization
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        else:
            if isinstance(original_image, torch.Tensor):
                img_display = original_image.cpu().numpy()
                if img_display.ndim == 4:
                    img_display = img_display[0]
                if img_display.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                    img_display = img_display.transpose(1, 2, 0)
            else:
                img_display = original_image
            
            # Normalize to [0, 1]
            if img_display.max() > 1:
                img_display = img_display / 255.0
        
        # Resize activation map to match image size
        h, w = img_display.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_display)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Activation map
        im = axes[1].imshow(cam_resized, cmap=cmap)
        axes[1].set_title(f'Activation Map (Class {pred_class})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        axes[2].imshow(img_display)
        axes[2].imshow(cam_resized, cmap=cmap, alpha=alpha)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def __del__(self):
        """Cleanup hooks"""
        self.remove_hooks()


def visualize_attention_maps(model, dataloader, num_samples=5, class_names=None, device='cuda'):
    """
    Visualize attention maps for random samples from dataloader
    
    Args:
        model: Model with attention mechanism (AttentionWrapper)
        dataloader: DataLoader with images
        num_samples: Number of samples to visualize
        class_names: List of class names for labels
        device: Device to run on
    
    Returns:
        fig: Matplotlib figure
    """
    model.eval()
    
    # Check if model has attention
    if not hasattr(model, 'get_attention_map'):
        print("⚠️ Model does not have attention mechanism")
        return None
    
    # Get samples
    images, labels = next(iter(dataloader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples].cpu().numpy()
    
    # Forward pass to generate attention maps
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().numpy()
    
    # Get attention maps
    attention_maps = model.get_attention_map()
    if attention_maps is None:
        print("⚠️ No attention maps available")
        return None
    
    # Visualize
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image (denormalize)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        # Attention map
        attn = attention_maps[i, 0].cpu().numpy()
        attn_resized = cv2.resize(attn, (img.shape[1], img.shape[0]))
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original\nTrue: {labels[i]}, Pred: {predictions[i]}')
        axes[i, 0].axis('off')
        
        im = axes[i, 1].imshow(attn_resized, cmap='jet')
        axes[i, 1].set_title('Attention Map')
        axes[i, 1].axis('off')
        plt.colorbar(im, ax=axes[i, 1])
        
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(attn_resized, cmap='jet', alpha=0.5)
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig


def debug_activations(model, image, layer_names=None):
    """
    Debug utility to inspect activations at multiple layers
    
    Args:
        model: Neural network model
        image: Input tensor
        layer_names: List of layer names to inspect (if None, shows all conv layers)
    
    Returns:
        activations_dict: Dictionary mapping layer names to activation tensors
    """
    model.eval()
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    model_to_search = model.backbone if hasattr(model, 'backbone') else model
    
    for name, module in model_to_search.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if layer_names is None or name in layer_names:
                hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        _ = model(image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Print summary
    print("\n📊 Activation Summary:")
    for name, act in activations.items():
        print(f"  {name}: {act.shape}")
    
    return activations


def plot_activation_statistics(activations_dict):
    """Plot statistics of activations across layers"""
    layer_names = list(activations_dict.keys())
    means = [activations_dict[name].mean().item() for name in layer_names]
    stds = [activations_dict[name].std().item() for name in layer_names]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean activations
    axes[0].bar(range(len(layer_names)), means)
    axes[0].set_xticks(range(len(layer_names)))
    axes[0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[0].set_title('Mean Activation per Layer')
    axes[0].set_ylabel('Mean Value')
    axes[0].grid(True, alpha=0.3)
    
    # Std activations
    axes[1].bar(range(len(layer_names)), stds)
    axes[1].set_xticks(range(len(layer_names)))
    axes[1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[1].set_title('Activation Std per Layer')
    axes[1].set_ylabel('Std Value')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
