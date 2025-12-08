"""
Outlier detection and data cleaning utilities
Implements ADVICE 05/12 - Inspect Outliers
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm


def find_high_loss_samples(model, dataloader, device, top_k=100):
    """
    Find samples with highest loss values
    
    Args:
        model: Trained model
        dataloader: DataLoader with dataset to inspect
        device: cuda or cpu
        top_k: Number of highest loss samples to return
    
    Returns:
        List of (index, loss_value, predicted_label, true_label)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss
    
    losses = []
    predictions = []
    labels = []
    indices = []
    
    with torch.no_grad():
        for batch_idx, (images, batch_labels) in enumerate(tqdm(dataloader, desc="Computing losses")):
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(images)
            batch_losses = criterion(outputs, batch_labels)
            _, preds = torch.max(outputs, 1)
            
            # Store results
            batch_size = images.size(0)
            batch_indices = range(batch_idx * dataloader.batch_size, 
                                 batch_idx * dataloader.batch_size + batch_size)
            
            losses.extend(batch_losses.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            indices.extend(batch_indices)
    
    # Create results dataframe
    results = pd.DataFrame({
        'index': indices,
        'loss': losses,
        'predicted': predictions,
        'true_label': labels,
        'correct': [p == l for p, l in zip(predictions, labels)]
    })
    
    # Sort by loss (highest first)
    results = results.sort_values('loss', ascending=False)
    
    return results.head(top_k)


def visualize_outliers(high_loss_df, dataset, save_path='outliers_visualization.png', 
                      num_samples=20):
    """
    Visualize high-loss samples to inspect mislabeled or corrupted data
    
    Args:
        high_loss_df: DataFrame from find_high_loss_samples()
        dataset: Original dataset to get images
        save_path: Where to save the visualization
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, len(high_loss_df))
    
    # Create grid
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (_, row) in enumerate(high_loss_df.head(num_samples).iterrows()):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        
        # Get image
        image_idx = int(row['index'])
        image, true_label = dataset[image_idx]
        
        # Convert tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
            # Denormalize if needed (assuming ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        
        # Add information
        loss = row['loss']
        pred = int(row['predicted'])
        true = int(row['true_label'])
        correct = row['correct']
        
        color = 'green' if correct else 'red'
        title = f"Loss: {loss:.3f}\nTrue: {true}, Pred: {pred}"
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Outlier visualization saved to {save_path}")
    plt.close()


def detect_corrupted_images(data_dir, image_column='Image', min_size=32):
    """
    Detect corrupted or invalid images in dataset
    
    Args:
        data_dir: Directory containing images
        image_column: Column name with image filenames
        min_size: Minimum valid image size
    
    Returns:
        List of corrupted image paths
    """
    corrupted = []
    
    image_dir = os.path.join(data_dir, "images")
    if not os.path.exists(image_dir):
        print(f"⚠️  Directory not found: {image_dir}")
        return corrupted
    
    image_files = os.listdir(image_dir)
    
    for img_file in tqdm(image_files, desc="Checking images"):
        img_path = os.path.join(image_dir, img_file)
        
        try:
            img = Image.open(img_path)
            img.verify()  # Check if file is corrupted
            
            # Re-open after verify (verify closes the file)
            img = Image.open(img_path)
            
            # Check size
            if img.size[0] < min_size or img.size[1] < min_size:
                corrupted.append(img_path)
                print(f"⚠️  Too small: {img_file} ({img.size})")
                continue
            
            # Check mode
            if img.mode not in ['RGB', 'L']:
                corrupted.append(img_path)
                print(f"⚠️  Invalid mode: {img_file} ({img.mode})")
                continue
                
        except Exception as e:
            corrupted.append(img_path)
            print(f"❌ Corrupted: {img_file} - {str(e)}")
    
    return corrupted


def find_label_errors(high_loss_df, threshold_percentile=95):
    """
    Find potential label errors based on consistently high loss
    
    Args:
        high_loss_df: DataFrame from find_high_loss_samples()
        threshold_percentile: Loss percentile threshold for suspicious samples
    
    Returns:
        DataFrame with suspicious samples
    """
    threshold = np.percentile(high_loss_df['loss'], threshold_percentile)
    
    suspicious = high_loss_df[
        (high_loss_df['loss'] > threshold) & 
        (~high_loss_df['correct'])
    ]
    
    print(f"\n🔍 Found {len(suspicious)} potentially mislabeled samples")
    print(f"   Loss threshold: {threshold:.3f} ({threshold_percentile}th percentile)")
    
    return suspicious


def create_outlier_report(model, dataloader, dataset, device, output_dir='outlier_analysis'):
    """
    Complete outlier analysis pipeline
    
    Args:
        model: Trained model
        dataloader: DataLoader
        dataset: Original dataset
        device: cuda or cpu
        output_dir: Directory to save results
    
    Returns:
        Dictionary with analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔍 Starting outlier analysis...")
    
    # 1. Find high loss samples
    print("\n1️⃣  Finding high-loss samples...")
    high_loss_df = find_high_loss_samples(model, dataloader, device, top_k=200)
    
    # 2. Detect label errors
    print("\n2️⃣  Detecting potential label errors...")
    suspicious = find_label_errors(high_loss_df, threshold_percentile=95)
    
    # 3. Visualize outliers
    print("\n3️⃣  Visualizing outliers...")
    viz_path = os.path.join(output_dir, 'high_loss_samples.png')
    visualize_outliers(high_loss_df, dataset, save_path=viz_path, num_samples=20)
    
    # 4. Save reports
    print("\n4️⃣  Saving reports...")
    high_loss_df.to_csv(os.path.join(output_dir, 'high_loss_samples.csv'), index=False)
    suspicious.to_csv(os.path.join(output_dir, 'suspicious_labels.csv'), index=False)
    
    # 5. Generate summary
    summary = {
        'total_samples': len(high_loss_df),
        'suspicious_labels': len(suspicious),
        'avg_loss_top100': high_loss_df.head(100)['loss'].mean(),
        'accuracy_top100': high_loss_df.head(100)['correct'].mean(),
        'suspicious_indices': suspicious['index'].tolist()
    }
    
    print("\n" + "="*50)
    print("📊 OUTLIER ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total high-loss samples analyzed: {summary['total_samples']}")
    print(f"Potentially mislabeled samples: {summary['suspicious_labels']}")
    print(f"Average loss (top 100): {summary['avg_loss_top100']:.3f}")
    print(f"Accuracy on top 100 highest-loss: {summary['accuracy_top100']:.1%}")
    print(f"\n💾 Results saved to: {output_dir}/")
    print("="*50)
    
    return summary
