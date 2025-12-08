"""
Data cleaning and preprocessing utilities
"""
import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


def create_clean_dataset(original_df, source_dir, dest_dir, dest_csv, threshold=0.005):
    """
    Filters the dataset and copies valid images/masks to a new folder.
    
    Args:
        original_df: Original DataFrame with labels
        source_dir: Source directory with train_data
        dest_dir: Destination directory for clean data
        dest_csv: Path to save cleaned CSV
        threshold: Minimum tissue percentage to keep sample
    
    Returns:
        DataFrame with cleaned data
    """
    print(f"🧹 Creating clean dataset in: {dest_dir}")
    
    # Create destination directories
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "masks"), exist_ok=True)
    
    clean_data = []
    issues = []
    
    for idx, row in tqdm(original_df.iterrows(), total=len(original_df), desc="Cleaning data"):
        img_name = row['Image']
        mask_name = row['Mask']
        label = row['Label']
        
        # Build file paths
        img_path = os.path.join(source_dir, "images", img_name)
        mask_path = os.path.join(source_dir, "masks", mask_name)
        
        # Check if files exist
        if not os.path.exists(img_path):
            issues.append(f"Missing image: {img_name}")
            continue
        
        if not os.path.exists(mask_path):
            issues.append(f"Missing mask: {mask_name}")
            continue
        
        try:
            # Load and validate mask
            mask = np.array(Image.open(mask_path))
            
            # Calculate tissue percentage
            tissue_pixels = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            tissue_percentage = tissue_pixels / total_pixels
            
            # Filter out samples with too little tissue
            if tissue_percentage < threshold:
                issues.append(f"Low tissue ({tissue_percentage:.4f}): {img_name}")
                continue
            
            # Copy files to clean directory
            dest_img_path = os.path.join(dest_dir, "images", img_name)
            dest_mask_path = os.path.join(dest_dir, "masks", mask_name)
            
            shutil.copy2(img_path, dest_img_path)
            shutil.copy2(mask_path, dest_mask_path)
            
            # Add to clean data
            clean_data.append({
                'Image': img_name,
                'Mask': mask_name,
                'Label': label,
                'TissuePercentage': tissue_percentage
            })
            
        except Exception as e:
            issues.append(f"Error processing {img_name}: {str(e)}")
            continue
    
    # Create cleaned DataFrame
    clean_df = pd.DataFrame(clean_data)
    clean_df.to_csv(dest_csv, index=False)
    
    print(f"\n✅ Clean dataset created:")
    print(f"   Original samples: {len(original_df)}")
    print(f"   Clean samples: {len(clean_df)}")
    print(f"   Removed: {len(original_df) - len(clean_df)}")
    print(f"   Issues found: {len(issues)}")
    
    if issues:
        print(f"\n⚠️  First 10 issues:")
        for issue in issues[:10]:
            print(f"   - {issue}")
    
    return clean_df


def analyze_dataset(df, title="Dataset Analysis"):
    """Quick dataset analysis"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    print(f"\nTotal samples: {len(df)}")
    print(f"\nClass distribution:")
    class_counts = df['Label'].value_counts().sort_index()
    for label, count in class_counts.items():
        print(f"   Class {label}: {count:4d} ({count/len(df)*100:5.2f}%)")
    
    if 'TissuePercentage' in df.columns:
        print(f"\nTissue coverage:")
        print(f"   Mean: {df['TissuePercentage'].mean():.4f}")
        print(f"   Min:  {df['TissuePercentage'].min():.4f}")
        print(f"   Max:  {df['TissuePercentage'].max():.4f}")
