"""
Custom Dataset and DataLoader utilities
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ImageNet normalization statistics (critical for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TissueDataset(Dataset):
    """Dataset for tissue classification"""
    
    def __init__(self, df, data_dir, transform=None, use_mask=True):
        """
        Args:
            df: DataFrame with Image, Mask, Label columns
            data_dir: Root directory containing images/ and masks/ folders
            transform: Optional transforms
            use_mask: Whether to apply mask to images
        """
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.use_mask = use_mask
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Handle both uppercase and lowercase column names
        img_name = row.get('Image', row.get('sample_index', row.get('image')))
        mask_name = row.get('Mask', row.get('mask', img_name))  # Use same name if mask col doesn't exist
        label_val = row.get('Label', row.get('label'))
        
        # Load image and mask - check if images/ subdirectory exists
        img_subdir_path = os.path.join(self.data_dir, "images", img_name)
        img_direct_path = os.path.join(self.data_dir, img_name)
        
        if os.path.exists(img_subdir_path):
            img_path = img_subdir_path
            mask_path = os.path.join(self.data_dir, "masks", mask_name)
        else:
            img_path = img_direct_path
            mask_path = img_direct_path  # Same file serves as both image and mask
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Apply mask if needed
        if self.use_mask:
            image_np = np.array(image)
            mask_np = np.array(mask)
            
            # Instead of setting background to black (0), set to ImageNet mean
            # This prevents distribution shift when masks aren't used
            imagenet_mean_rgb = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.uint8)
            
            # Create background with ImageNet mean color
            masked_image = np.where(
                mask_np[:, :, np.newaxis] > 0,  # Where mask is active
                image_np,  # Keep original image
                imagenet_mean_rgb  # Use ImageNet mean for background
            )
            image = Image.fromarray(masked_image.astype(np.uint8))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Handle label (convert string to int if needed)
        if isinstance(label_val, str):
            # Map class names to integers (handle both HER2(+) and HER2 positive)
            class_mapping = {
                'Triple negative': 0,
                'Luminal A': 1,
                'Luminal B': 2,
                'HER2 positive': 3,
                'HER2(+)': 3  # Alternative format in dataset
            }
            label = torch.tensor(class_mapping.get(label_val, 0), dtype=torch.long)
        else:
            label = torch.tensor(label_val, dtype=torch.long)
        
        return image, label


def get_transforms(img_size=256, augment=True, use_imagenet_norm=True,
                   use_automated_aug=False, auto_method='randaugment', auto_n=2, auto_m=9):
    """Get train and validation transforms
    
    Args:
        img_size: Target image size
        augment: Whether to apply data augmentation (training)
        use_imagenet_norm: Use ImageNet normalization (required for pretrained models)
        use_automated_aug: Use automated augmentation (RandAugment, TrivialAugment)
        auto_method: Automated augmentation method ('randaugment' or 'trivialaugment')
        auto_n: Number of augmentation operations (for RandAugment)
        auto_m: Magnitude of augmentations 0-10 (for RandAugment)
    
    Uses manual augmentation from original notebooks (challenge-2-preprocessing.ipynb):
    - ColorJitter, RandomRotation, RandomHorizontalFlip, RandomErasing
    Or automated augmentation from ADVICE 06/12
    """
    
    # Normalization (ImageNet stats for transfer learning - from lectures)
    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ) if use_imagenet_norm else transforms.Lambda(lambda x: x)
    
    if augment:
        if use_automated_aug:
            # Automated augmentation - ADVICE 06/12: "Let the policy emerge from the struggle"
            # get_automated_augmentation imported at module level
            auto_aug = get_automated_augmentation(method=auto_method, n=auto_n, m=auto_m)
            
            transform_list = [
                transforms.Resize((img_size, img_size)),
                auto_aug,
                transforms.ToTensor(),
                normalize
            ]
        else:
            # Manual augmentation from original challenge-2-preprocessing.ipynb
            transform_list = [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=15,                # Rotation
                    translate=(0.1, 0.1),      # Translation
                    scale=(0.9, 1.1)           # Zoom
                ),
                transforms.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.2
                ),
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.3,                     # Probability
                    scale=(0.02, 0.15),        # Size of erased area
                    ratio=(0.3, 3.3),          # Aspect ratio
                    value='random'             # Fill with random values
                ),
                normalize
            ]
        
        train_transform = transforms.Compose(transform_list)
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def create_dataloaders(train_df, val_df, data_dir, config, use_imagenet_norm=True):
    """Create optimized train and validation dataloaders
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        data_dir: Root data directory
        config: Configuration object
        use_imagenet_norm: Use ImageNet normalization (True for pretrained models)
    
    Returns:
        train_loader, val_loader, num_classes
    """
    
    # Check for automated augmentation settings
    use_automated_aug = getattr(config, 'USE_AUTOMATED_AUG')
    auto_method = getattr(config, 'AUTOMATED_AUG_METHOD')
    auto_n = getattr(config, 'RANDAUGMENT_N')
    auto_m = getattr(config, 'RANDAUGMENT_M')
    
    train_transform, val_transform = get_transforms(
        img_size=config.IMG_SIZE,
        augment=True,
        use_imagenet_norm=use_imagenet_norm,
        use_automated_aug=use_automated_aug,
        auto_method=auto_method,
        auto_n=auto_n,
        auto_m=auto_m
    )
    
    train_dataset = TissueDataset(train_df, data_dir, transform=train_transform)
    val_dataset = TissueDataset(val_df, data_dir, transform=val_transform)
    
    # Auto-detect number of classes from data
    label_col = 'label' if 'label' in train_df.columns else 'Label'
    num_classes = train_df[label_col].nunique()
    print(f"✅ Detected {num_classes} classes in dataset")
    
    # Optimized DataLoader settings from AN2DL lectures
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        drop_last=True  # Drop incomplete batch for training stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
    )
    
    return train_loader, val_loader, num_classes


class TestDataset(Dataset):
    """Dataset for test images (no labels)"""
    
    def __init__(self, test_dir, transform=None, file_pattern='img_', use_mask=True):
        """
        Args:
            test_dir: Directory containing test images
            transform: Optional transforms to apply
            file_pattern: Pattern to filter files (default: 'img_' to exclude masks)
            use_mask: Whether to apply mask to images if available
        """
        self.test_dir = test_dir
        self.transform = transform
        self.use_mask = use_mask
        
        # Get all image files
        all_files = [f for f in os.listdir(test_dir) 
                    if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Filter by pattern (e.g., only img_*.png, not mask_*.png)
        self.image_files = sorted([f for f in all_files if f.startswith(file_pattern)])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Apply mask if available and use_mask=True
        if self.use_mask:
            # Try to find corresponding mask
            mask_name = img_name.replace('img_', 'mask_')
            mask_path = os.path.join(self.test_dir, mask_name)
            
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                
                # Apply mask - set background to ImageNet mean instead of black
                image_np = np.array(image)
                mask_np = np.array(mask)
                
                # ImageNet mean RGB values (unnormalized)
                imagenet_mean_rgb = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.uint8)
                
                # Create background with ImageNet mean color
                masked_image = np.where(
                    mask_np[:, :, np.newaxis] > 0,  # Where mask is active
                    image_np,  # Keep original image
                    imagenet_mean_rgb  # Use ImageNet mean for background
                )
                image = Image.fromarray(masked_image.astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_name  # Return filename for submission
