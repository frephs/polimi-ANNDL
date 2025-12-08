"""
Automated augmentation strategies
Implements ADVICE 06/12 - Automated Augmentation
"""
import torch
import torch.nn as nn
from torchvision import transforms
import random
import numpy as np


class RandAugment:
    """
    RandAugment: Practical automated data augmentation
    Paper: https://arxiv.org/abs/1909.13719
    
    Randomly select N augmentation transformations from a set and apply them
    with magnitude M.
    """
    def __init__(self, n=2, m=9):
        """
        Args:
            n: Number of augmentation transformations to apply sequentially
            m: Magnitude for all augmentation operations (0-10 scale)
        """
        self.n = n
        self.m = m
        self.augment_list = [
            self.auto_contrast,
            self.equalize,
            self.rotate,
            self.solarize,
            self.color,
            self.posterize,
            self.contrast,
            self.brightness,
            self.sharpness,
            self.shear_x,
            self.shear_y,
            self.translate_x,
            self.translate_y
        ]
    
    def __call__(self, img):
        """Apply N random augmentations"""
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            img = op(img)
        return img
    
    def auto_contrast(self, img):
        return transforms.functional.autocontrast(img)
    
    def equalize(self, img):
        return transforms.functional.equalize(img)
    
    def rotate(self, img):
        magnitude = (self.m / 10) * 30  # Max 30 degrees
        angle = random.uniform(-magnitude, magnitude)
        return transforms.functional.rotate(img, angle)
    
    def solarize(self, img):
        magnitude = int((self.m / 10) * 256)
        return transforms.functional.solarize(img, threshold=256 - magnitude)
    
    def color(self, img):
        magnitude = (self.m / 10) * 0.9 + 0.1  # 0.1 to 1.0
        return transforms.functional.adjust_saturation(img, magnitude)
    
    def posterize(self, img):
        magnitude = int((self.m / 10) * 4)  # Reduce bits by 0-4
        bits = 8 - magnitude
        return transforms.functional.posterize(img, bits)
    
    def contrast(self, img):
        magnitude = (self.m / 10) * 0.9 + 0.1
        return transforms.functional.adjust_contrast(img, magnitude)
    
    def brightness(self, img):
        magnitude = (self.m / 10) * 0.9 + 0.1
        return transforms.functional.adjust_brightness(img, magnitude)
    
    def sharpness(self, img):
        magnitude = (self.m / 10) * 0.9 + 0.1
        return transforms.functional.adjust_sharpness(img, magnitude)
    
    def shear_x(self, img):
        magnitude = (self.m / 10) * 0.3  # Max shear 0.3
        return transforms.functional.affine(img, angle=0, translate=[0, 0], 
                                           scale=1, shear=[magnitude * 180 / np.pi, 0])
    
    def shear_y(self, img):
        magnitude = (self.m / 10) * 0.3
        return transforms.functional.affine(img, angle=0, translate=[0, 0], 
                                           scale=1, shear=[0, magnitude * 180 / np.pi])
    
    def translate_x(self, img):
        magnitude = int((self.m / 10) * img.size[0] * 0.3)  # Max 30% of width
        return transforms.functional.affine(img, angle=0, translate=[magnitude, 0], 
                                           scale=1, shear=[0, 0])
    
    def translate_y(self, img):
        magnitude = int((self.m / 10) * img.size[1] * 0.3)
        return transforms.functional.affine(img, angle=0, translate=[0, magnitude], 
                                           scale=1, shear=[0, 0])


class TrivialAugmentWide:
    """
    TrivialAugment: Simple and effective augmentation
    Paper: https://arxiv.org/abs/2103.10158
    
    Randomly selects ONE augmentation per image with random magnitude.
    Often outperforms more complex strategies.
    """
    def __init__(self):
        self.augment_list = [
            ('Identity', 0, 0),
            ('AutoContrast', 0, 0),
            ('Equalize', 0, 0),
            ('Rotate', -30, 30),
            ('Solarize', 0, 256),
            ('Color', 0.1, 1.9),
            ('Posterize', 2, 8),
            ('Contrast', 0.1, 1.9),
            ('Brightness', 0.1, 1.9),
            ('Sharpness', 0.1, 1.9),
            ('ShearX', -0.3, 0.3),
            ('ShearY', -0.3, 0.3),
            ('TranslateX', -0.3, 0.3),
            ('TranslateY', -0.3, 0.3)
        ]
    
    def __call__(self, img):
        """Randomly select and apply ONE augmentation"""
        op_name, min_val, max_val = random.choice(self.augment_list)
        magnitude = random.uniform(min_val, max_val) if max_val > min_val else 0
        
        if op_name == 'Identity':
            return img
        elif op_name == 'AutoContrast':
            return transforms.functional.autocontrast(img)
        elif op_name == 'Equalize':
            return transforms.functional.equalize(img)
        elif op_name == 'Rotate':
            return transforms.functional.rotate(img, magnitude)
        elif op_name == 'Solarize':
            return transforms.functional.solarize(img, int(magnitude))
        elif op_name == 'Color':
            return transforms.functional.adjust_saturation(img, magnitude)
        elif op_name == 'Posterize':
            return transforms.functional.posterize(img, int(magnitude))
        elif op_name == 'Contrast':
            return transforms.functional.adjust_contrast(img, magnitude)
        elif op_name == 'Brightness':
            return transforms.functional.adjust_brightness(img, magnitude)
        elif op_name == 'Sharpness':
            return transforms.functional.adjust_sharpness(img, magnitude)
        elif op_name == 'ShearX':
            return transforms.functional.affine(img, angle=0, translate=[0, 0], 
                                               scale=1, shear=[magnitude * 180 / np.pi, 0])
        elif op_name == 'ShearY':
            return transforms.functional.affine(img, angle=0, translate=[0, 0], 
                                               scale=1, shear=[0, magnitude * 180 / np.pi])
        elif op_name == 'TranslateX':
            pixels = int(magnitude * img.size[0])
            return transforms.functional.affine(img, angle=0, translate=[pixels, 0], 
                                               scale=1, shear=[0, 0])
        elif op_name == 'TranslateY':
            pixels = int(magnitude * img.size[1])
            return transforms.functional.affine(img, angle=0, translate=[0, pixels], 
                                               scale=1, shear=[0, 0])
        return img


def get_automated_augmentation(method='randaugment', **kwargs):
    """
    Factory function to get automated augmentation
    
    Args:
        method: 'randaugment', 'trivialaugment', or 'autoaugment'
        **kwargs: Method-specific parameters
    
    Returns:
        Augmentation transform
    """
    if method == 'randaugment':
        n = kwargs.get('n', 2)  # Number of ops
        m = kwargs.get('m', 9)  # Magnitude
        return RandAugment(n=n, m=m)
    
    elif method == 'trivialaugment':
        return TrivialAugmentWide()
    
    elif method == 'autoaugment':
        # Use PyTorch's built-in AutoAugment
        from torchvision.transforms import AutoAugment, AutoAugmentPolicy
        policy = kwargs.get('policy', AutoAugmentPolicy.IMAGENET)
        return AutoAugment(policy=policy)
    
    else:
        raise ValueError(f"Method {method} not supported. Choose from: "
                        "randaugment, trivialaugment, autoaugment")
