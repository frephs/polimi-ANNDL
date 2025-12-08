"""
Advanced augmentation techniques: CutMix, MixUp
Based on PERFORMANCE_ROADMAP recommendations
"""
import torch
import numpy as np


class CutMix:
    """CutMix augmentation for training
    
    Cuts a random region from one image and pastes it into another,
    with labels mixed proportionally to the area.
    
    Paper: CutMix: Regularization Strategy to Train Strong Classifiers
    https://arxiv.org/abs/1905.04899
    """
    def __init__(self, alpha=1.0, prob=0.5):
        """
        Args:
            alpha: Beta distribution parameter (higher = more mixing)
            prob: Probability of applying CutMix
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch_images, batch_labels):
        """Apply CutMix to a batch
        
        Args:
            batch_images: Tensor of shape (B, C, H, W)
            batch_labels: Tensor of shape (B,) with class indices
            
        Returns:
            mixed_images: CutMix augmented images
            labels_a: Original labels
            labels_b: Mixed labels
            lam: Mixing coefficient
        """
        if np.random.rand() > self.prob:
            return batch_images, batch_labels, batch_labels, 1.0
        
        batch_size = batch_images.size(0)
        indices = torch.randperm(batch_size).to(batch_images.device)
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Generate random box
        _, _, H, W = batch_images.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Random center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Cut and paste
        batch_images[:, :, bby1:bby2, bbx1:bbx2] = batch_images[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return batch_images, batch_labels, batch_labels[indices], lam


class MixUp:
    """MixUp augmentation for training
    
    Linearly interpolates two images and their labels.
    
    Paper: mixup: Beyond Empirical Risk Minimization
    https://arxiv.org/abs/1710.09412
    """
    def __init__(self, alpha=0.2, prob=0.5):
        """
        Args:
            alpha: Beta distribution parameter (lower = less mixing for medical images)
            prob: Probability of applying MixUp
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch_images, batch_labels):
        """Apply MixUp to a batch
        
        Args:
            batch_images: Tensor of shape (B, C, H, W)
            batch_labels: Tensor of shape (B,) with class indices
            
        Returns:
            mixed_images: MixUp augmented images
            labels_a: Original labels
            labels_b: Mixed labels
            lam: Mixing coefficient
        """
        if np.random.rand() > self.prob:
            return batch_images, batch_labels, batch_labels, 1.0
        
        batch_size = batch_images.size(0)
        indices = torch.randperm(batch_size).to(batch_images.device)
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_images = lam * batch_images + (1 - lam) * batch_images[indices]
        
        return mixed_images, batch_labels, batch_labels[indices], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss function for MixUp/CutMix
    
    Args:
        criterion: Loss function (e.g., CrossEntropyLoss)
        pred: Model predictions
        y_a: Original labels
        y_b: Mixed labels
        lam: Mixing coefficient
        
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
