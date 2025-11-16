"""
Custom loss functions for training.

Implements label smoothing and other advanced loss techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        weight: Class weights for handling imbalance (optional)
        reduction: 'mean', 'sum', or 'none'
    
    """
    
    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (logits) of shape (batch_size, num_classes)
            target: Ground truth labels of shape (batch_size,) with class indices
        
        Returns:
            Smoothed cross-entropy loss
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            # Create one-hot encoding
            num_classes = pred.size(-1)
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Apply class weights if provided
        if self.weight is not None:
            # Expand weights to match true_dist shape
            weight_expanded = self.weight.unsqueeze(0).expand_as(true_dist)
            loss = -torch.sum(true_dist * pred * weight_expanded, dim=-1)
        else:
            loss = -torch.sum(true_dist * pred, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal loss focuses training on hard examples by down-weighting easy examples.
    Particularly useful when combined with class weights.
    
    Args:
        alpha: Weighting factor for class balance
        gamma: Focusing parameter
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (logits) of shape (batch_size, num_classes)
            target: Ground truth labels of shape (batch_size,)
        
        Returns:
            Focal loss
        """
        # Get log probabilities
        log_prob = F.log_softmax(pred, dim=-1)
        prob = torch.exp(log_prob)
        
        # Gather the probabilities for the correct class
        ce_loss = F.nll_loss(log_prob, target, reduction='none')
        p_t = prob.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        focal_term = (1 - p_t) ** self.gamma
        loss = focal_term * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # alpha is a tensor of class weights
                alpha_t = self.alpha.gather(0, target)
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
