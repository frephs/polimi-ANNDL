"""
Model architectures with attention mechanisms - config-driven
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights,
    EfficientNet_B0_Weights, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights,
    MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights,
    VGG16_Weights, DenseNet121_Weights,
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights
)


class MaskedSpatialAttention(nn.Module):
    """Spatial attention mechanism for feature maps"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate attention map
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        # Apply attention
        return x * attention, attention


class ChannelAttention(nn.Module):
    """Channel attention mechanism (Squeeze-and-Excitation)"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionWrapper(nn.Module):
    """Wraps model with attention mechanisms
    
    NOTE: Attention mechanisms require 4D feature maps (B, C, H, W).
    For models that internally pool features, attention is disabled automatically.
    """
    def __init__(self, backbone, num_classes=8, dropout=0.5, 
                 use_spatial_attention=True, use_channel_attention=True):
        super().__init__()
        self.backbone = backbone
        self.use_spatial_attention = use_spatial_attention
        self.use_channel_attention = use_channel_attention
        
        # Get feature dimension and remove original classifier
        if hasattr(backbone, 'fc'):
            # ResNet-style: has avgpool and fc
            self.feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            # Remove avgpool to get feature maps for attention
            if hasattr(backbone, 'avgpool'):
                backbone.avgpool = nn.Identity()
                self.needs_pooling = True
            else:
                self.needs_pooling = False
        elif hasattr(backbone, 'classifier'):
            # EfficientNet/MobileNet-style: has classifier
            if isinstance(backbone.classifier, nn.Sequential):
                # Find the last Linear layer in the sequential
                for layer in reversed(backbone.classifier):
                    if isinstance(layer, nn.Linear):
                        self.feature_dim = layer.in_features
                        break
            else:
                self.feature_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
            
            # For EfficientNet/MobileNet, pooling happens inside features module
            # We can't easily remove it, so we'll detect 2D output and handle it
            self.needs_pooling = False
        
        # Attention modules operate on feature maps (will be disabled if output is 2D)
        if use_spatial_attention:
            self.spatial_attention = MaskedSpatialAttention(self.feature_dim)
        if use_channel_attention:
            self.channel_attention = ChannelAttention(self.feature_dim)
        
        # Adaptive classifier - dimension will be adjusted on first forward pass if needed
        self.classifier = None
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.last_attention_map = None
        
    def forward(self, x):
        features = self.backbone(x)
        
        # Lazy initialization of classifier based on actual feature shape
        if self.classifier is None:
            if len(features.shape) == 4:
                # 4D feature maps - attention can be applied
                actual_dim = features.size(1)  # Channel dimension
            else:
                # 2D features - already pooled internally
                actual_dim = features.size(1)
                # Disable attention for 2D features
                self.use_spatial_attention = False
                self.use_channel_attention = False
            
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(actual_dim),
                nn.Dropout(self.dropout),
                nn.Linear(actual_dim, self.num_classes)
            ).to(features.device)
        
        # Handle case where backbone outputs 2D (some models pool internally)
        if len(features.shape) == 2:
            # Features are already pooled, skip attention
            return self.classifier(features)
        
        # Now we have 4D feature maps (batch, channels, height, width)
        if self.use_channel_attention:
            features = self.channel_attention(features)
        
        if self.use_spatial_attention:
            features, attention_map = self.spatial_attention(features)
            self.last_attention_map = attention_map.detach()
        
        # Global pooling
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        
        return self.classifier(features)
    
    def get_attention_map(self):
        """Get last attention map for visualization"""
        return self.last_attention_map


def get_model(config):
    """Build model from config
    
    Supported models:
        resnet18/34/50/101/152, efficientnet_b0/v2_s/v2_m, mobilenet_v3_large/small,
        vgg16, densenet121, convnext_tiny/small
    """
    model_name = config.MODEL_NAME.lower()
    num_classes = config.NUM_CLASSES
    pretrained = getattr(config, 'USE_PRETRAINED')
    use_attention = getattr(config, 'USE_ATTENTION')
    dropout = getattr(config, 'DROPOUT')
    norm_type = getattr(config, 'NORM_TYPE')
    batch_size = getattr(config, 'BATCH_SIZE')
    
    # Model registry
    models_registry = {
        'resnet18': (models.resnet18, ResNet18_Weights.IMAGENET1K_V1, 'fc'),
        'resnet34': (models.resnet34, ResNet34_Weights.IMAGENET1K_V1, 'fc'),
        'resnet50': (models.resnet50, ResNet50_Weights.IMAGENET1K_V2, 'fc'),
        'resnet101': (models.resnet101, ResNet101_Weights.IMAGENET1K_V2, 'fc'),
        'resnet152': (models.resnet152, ResNet152_Weights.IMAGENET1K_V2, 'fc'),
        'efficientnet_b0': (models.efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, 'classifier'),
        'efficientnet_v2_s': (models.efficientnet_v2_s, EfficientNet_V2_S_Weights.IMAGENET1K_V1, 'classifier'),
        'efficientnet_v2_m': (models.efficientnet_v2_m, EfficientNet_V2_M_Weights.IMAGENET1K_V1, 'classifier'),
        'mobilenet_v3_large': (models.mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2, 'classifier'),
        'mobilenet_v3_small': (models.mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1, 'classifier'),
        'vgg16': (models.vgg16_bn, VGG16_Weights.IMAGENET1K_V1, 'classifier'),
        'densenet121': (models.densenet121, DenseNet121_Weights.IMAGENET1K_V1, 'classifier'),
        'convnext_tiny': (models.convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 'classifier'),
        'convnext_small': (models.convnext_small, ConvNeXt_Small_Weights.IMAGENET1K_V1, 'classifier'),
    }
    
    if model_name not in models_registry:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(models_registry.keys())}")
    
    model_fn, weights, classifier_name = models_registry[model_name]
    model = model_fn(weights=weights if pretrained else None)
    
    # Replace classifier
    if classifier_name == 'fc':
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    elif classifier_name == 'classifier':
        if isinstance(model.classifier, nn.Sequential):
            # Find the last Linear layer in the sequential
            for layer in reversed(model.classifier):
                if isinstance(layer, nn.Linear):
                    num_features = layer.in_features
                    break
            model.classifier[-1] = nn.Linear(num_features, num_classes)
            # Insert batch norm and dropout before final layer
            model.classifier = nn.Sequential(
                *list(model.classifier[:-1]),
                nn.BatchNorm1d(num_features),
                nn.Dropout(dropout),
                model.classifier[-1]
            )
        else:
            num_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes)
            )
    
    # Wrap with attention if requested
    if use_attention:
        model = AttentionWrapper(model, num_classes=num_classes, dropout=dropout)
    
    # Apply normalization strategy - ADVICE 04/12
    # "Do not force the ocean's rules upon a cup of water"
    if norm_type == 'group' and batch_size < 16:
        # replace_batchnorm_with_groupnorm imported at module level
        model = replace_batchnorm_with_groupnorm(model, num_groups=32)
        print(f"   ✅ Applied GroupNorm (batch_size={batch_size} < 16)")
    elif norm_type == 'layer':
        # replace_batchnorm_with_layernorm imported at module level
        model = replace_batchnorm_with_layernorm(model)
        print(f"   ✅ Applied LayerNorm2d")
    
    # Freeze backbone if transfer learning
    if getattr(config, 'FREEZE_BACKBONE'):
        freeze_backbone(model, freeze=True)
    
    params = count_parameters(model)
    print(f"✅ Model: {model_name} | Params: {params:,} | Pretrained: {pretrained}")
    
    return model


def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable


def freeze_backbone(model, freeze=True):
    """Freeze/unfreeze backbone for transfer learning"""
    if isinstance(model, AttentionWrapper):
        target = model.backbone
    else:
        target = model
    
    # Freeze all backbone parameters
    for param in target.parameters():
        param.requires_grad = not freeze
    
    # Keep classifier trainable
    if isinstance(model, AttentionWrapper):
        for param in model.classifier.parameters():
            param.requires_grad = True
        if model.use_spatial_attention:
            for param in model.spatial_attention.parameters():
                param.requires_grad = True
        if model.use_channel_attention:
            for param in model.channel_attention.parameters():
                param.requires_grad = True
    else:
        if hasattr(target, 'fc'):
            for param in target.fc.parameters():
                param.requires_grad = True
        elif hasattr(target, 'classifier'):
            for param in target.classifier.parameters():
                param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"{'🔒 Frozen' if freeze else '🔓 Unfrozen'} backbone: {trainable:,} / {total:,} params trainable")
    
    return model
