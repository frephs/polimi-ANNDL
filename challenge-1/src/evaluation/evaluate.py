"""
Model evaluation utilities
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Literal, Optional


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    task: Literal['classification', 'regression'] = 'classification'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        task: Type of task ('classification' or 'regression')
        
    Returns:
        Tuple of (predictions, targets)
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            if task == 'classification':
                predictions = outputs.argmax(dim=1)
            else:  # regression
                predictions = outputs
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    return predictions, targets


def evaluate_and_report(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    dataset_name: str = "Validation",
    task: Literal['classification', 'regression'] = 'classification',
    class_names: Optional[list] = None,
    plot_cm: bool = True
) -> Dict[str, float]:
    """
    Evaluate model and print/plot results.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        dataset_name: Name of the dataset (for display)
        task: Type of task ('classification' or 'regression')
        class_names: Optional list of class names (for confusion matrix)
        plot_cm: Whether to plot confusion matrix (classification only)
        
    Returns:
        Dictionary of metrics
    """
    from ..training.metrics import calculate_metrics, plot_confusion_matrix, print_metrics_summary
    
    # Get predictions
    predictions, targets = evaluate_model(model, data_loader, device, task)
    
    # Calculate metrics
    metrics = calculate_metrics(targets, predictions, task)
    
    # Print summary
    print_metrics_summary(metrics, task, dataset_name)
    
    # Plot confusion matrix for classification
    if task == 'classification' and plot_cm:
        plot_confusion_matrix(targets, predictions, class_names)
    
    return metrics


def export_evaluation_results(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str
) -> None:
    """
    Export evaluation results to a CSV file.
    
    Args:
        predictions: Model predictions
        targets: True labels
        output_path: Path to save the CSV file
    """
    import pandas as pd
    
    results_df = pd.DataFrame({
        'TrueLabel': targets,
        'PredictedLabel': predictions
    })
    
    results_df.to_csv(output_path, index=True)