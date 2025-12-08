"""
Evaluation and inference utilities
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)


def predict_with_tta(model, image, device, num_augmentations=5, img_size=256):
    """Test-Time Augmentation for robust predictions
    
    Averages predictions over multiple augmented versions of the same image.
    This improves robustness and typically adds +1-2% accuracy.
    
    Args:
        model: Trained model
        image: Single image tensor (C, H, W) or (B, C, H, W)
        device: Device to run on
        num_augmentations: Number of augmented predictions to average
        img_size: Image size for transforms
        
    Returns:
        averaged_logits: Mean logits across all augmentations
        averaged_probs: Mean probabilities across all augmentations
    """
    model.eval()
    
    # Handle batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # TTA transforms (geometric only, no color changes)
    tta_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10)
    ])
    
    predictions = []
    
    with torch.no_grad():
        # Original prediction
        img_device = image.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            pred = model(img_device)
        predictions.append(pred)
        
        # Augmented predictions
        for _ in range(num_augmentations - 1):
            aug_image = tta_transforms(image).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                pred = model(aug_image)
            predictions.append(pred)
    
    # Average predictions
    avg_logits = torch.stack(predictions).mean(dim=0)
    avg_probs = F.softmax(avg_logits, dim=1)
    
    return avg_logits, avg_probs


def evaluate_model(model, data_loader, device, class_names=None):
    """Comprehensive model evaluation
    
    Args:
        model: Trained model
        data_loader: Test/validation dataloader
        device: Device to run on
        class_names: List of class names for reporting
    
    Returns:
        Dictionary with all metrics and predictions
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    
    # Print report
    print("\n📊 Evaluation Results:")
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1 Score:  {results['f1']:.4f}")
    
    if class_names:
        print("\n📋 Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return results


def plot_confusion_matrix(cm, class_names=None, figsize=(10, 8), save_path=None):
    """Plot confusion matrix heatmap
    
    Args:
        cm: Confusion matrix from sklearn
        class_names: List of class names
        figsize: Figure size
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names or range(cm.shape[0]),
        yticklabels=class_names or range(cm.shape[0]),
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history_df, save_path=None):
    """Plot training history curves
    
    Args:
        history_df: DataFrame with training history
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history_df['epoch'], history_df['train_loss'], label='Train', marker='o')
    axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Val', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history_df['epoch'], history_df['train_acc'], label='Train', marker='o')
    axes[1].plot(history_df['epoch'], history_df['val_acc'], label='Val', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[2].plot(history_df['epoch'], history_df['train_f1'], label='Train', marker='o')
    axes[2].plot(history_df['epoch'], history_df['val_f1'], label='Val', marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def predict_single_image(model, image_path, transform, device, class_names=None):
    """Predict on a single image
    
    Args:
        model: Trained model
        image_path: Path to image file
        transform: Transforms to apply
        device: Device to run on
        class_names: List of class names
    
    Returns:
        Dictionary with prediction, probability, and class name
    """
    from PIL import Image
    
    model.eval()
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            output = model(image_tensor)
        
        probs = torch.softmax(output, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    result = {
        'prediction': pred_idx,
        'confidence': confidence,
        'class_name': class_names[pred_idx] if class_names else str(pred_idx),
        'all_probabilities': probs[0].cpu().numpy()
    }
    
    return result


def create_submission(model, test_dir, config, device, output_path='submission.csv', 
                     class_to_label=None, use_mask=True, use_tta=False, tta_num_aug=5):
    """Create submission file for test set
    
    Args:
        model: Trained model
        test_dir: Directory containing test images
        config: Config object with IMG_SIZE and normalization settings
        device: Device to run on
        output_path: Path to save submission CSV
        class_to_label: Optional dict mapping class indices to label names
        use_mask: Whether to apply masks if available (default: True)
        use_tta: Whether to use Test-Time Augmentation (default: False, +1-2% accuracy)
        tta_num_aug: Number of TTA augmentations if use_tta=True
    
    Returns:
        submission_df: DataFrame with predictions
    """
    import pandas as pd
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    # TestDataset and get_transforms imported at module level
    
    # Create test dataset with masks if available
    _, test_transform = get_transforms(
        img_size=config.IMG_SIZE,
        augment=False,
        use_imagenet_norm=config.USE_IMAGENET_NORM
    )
    
    test_dataset = TestDataset(
        test_dir=test_dir,
        transform=test_transform,
        file_pattern='img_',
        use_mask=use_mask
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    model.eval()
    all_preds = []
    all_filenames = []
    
    if use_tta:
        print(f"🔄 Using Test-Time Augmentation ({tta_num_aug} augmentations per image)")
        # TTA: Process images individually for augmentation
        for images, filenames in tqdm(test_loader, desc="Generating predictions with TTA"):
            batch_preds = []
            for img in images:
                # Apply TTA to each image
                _, avg_probs = predict_with_tta(
                    model, img, device, 
                    num_augmentations=tta_num_aug,
                    img_size=config.IMG_SIZE
                )
                pred = avg_probs.argmax(dim=1).item()
                batch_preds.append(pred)
            
            all_preds.extend(batch_preds)
            all_filenames.extend(filenames)
    else:
        # Standard prediction (faster)
        with torch.no_grad():
            for images, filenames in tqdm(test_loader, desc="Generating predictions"):
                images = images.to(device)
                
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs = model(images)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_filenames.extend(filenames)
    
    # Convert class numbers to labels
    # Default class mapping (standard for this competition)
    if class_to_label is None:
        class_to_label = {
            0: 'Triple negative',
            1: 'Luminal A',
            2: 'Luminal B',
            3: 'HER2(+)'
        }
    
    predictions = [class_to_label[pred] for pred in all_preds]
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'sample_index': all_filenames,
        'label': predictions
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Submission saved to {output_path}")
    print(f"   Rows: {len(submission_df)}")
    print(f"   Masks applied: {use_mask}")
    print(f"   TTA enabled: {use_tta}")
    if use_tta:
        print(f"   TTA augmentations: {tta_num_aug}")
    
    if class_to_label:
        print(f"\nLabel distribution:")
        print(submission_df['label'].value_counts())
    
    return submission_df