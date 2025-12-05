"""
Evaluation metrics and visualization utilities for classification and regression
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from typing import Dict, List, Optional, Literal


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: Literal['classification', 'regression'] = 'classification'
) -> Dict[str, float]:
    """
    Calculate metrics for classification or regression tasks.
    
    Args:
        y_true: True labels (classification) or values (regression)
        y_pred: Predicted labels (classification) or values (regression)
        task: Type of task ('classification' or 'regression')
        
    Returns:
        Dictionary containing task-specific metrics
    """
    if task == 'classification':
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0, ),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }
    elif task == 'regression':
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    else:
        raise ValueError(f"Unknown task: {task}")


def plot_training_history(
    history: Dict[str, List[float]],
    task: Literal['classification', 'regression'] = 'classification',
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history (loss and metrics).
    
    Args:
        history: Dictionary containing training history
        task: Type of task ('classification' or 'regression')
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training loss', alpha=0.3, 
             color='#ff7f0e', linestyle='--')
    ax1.plot(history['val_loss'], label='Validation loss', alpha=0.9, 
             color='#ff7f0e')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot metric
    if task == 'classification':
        metric_key = 'f1'
        metric_name = 'F1 Score'
    else:  # regression
        metric_key = 'r2'
        metric_name = 'R² Score'
    
    train_metric_key = f'train_{metric_key}'
    val_metric_key = f'val_{metric_key}'
    
    if train_metric_key in history and val_metric_key in history:
        ax2.plot(history[train_metric_key], label=f'Training {metric_name}', alpha=0.3, 
                 color='#ff7f0e', linestyle='--')
        ax2.plot(history[val_metric_key], label=f'Validation {metric_name}', alpha=0.9, 
                 color='#ff7f0e')
        ax2.set_title(metric_name)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        save_path: Optional path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Create labels
    labels = np.array([f"{num}" for num in cm.flatten()]).reshape(cm.shape)
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    if class_names is not None:
        sns.heatmap(
            cm,
            annot=labels,
            fmt='',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
    else:
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot predicted vs actual values for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    
    # Scatter plot: Predicted vs Actual
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predicted vs Actual Values')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Residual plot
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_metrics_summary(
    metrics: Dict[str, float],
    task: Literal['classification', 'regression'] = 'classification',
    dataset_name: str = "Dataset"
) -> None:
    """
    Print a summary of evaluation metrics.
    
    Args:
        metrics: Dictionary containing metrics
        task: Type of task ('classification' or 'regression')
        dataset_name: Name of the dataset being evaluated
    """
    print(f"\n{dataset_name} Metrics:")
    print("=" * 50)
    
    if task == 'classification':
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
    else:  # regression
        print(f"MSE:       {metrics['mse']:.4f}")
        print(f"RMSE:      {metrics['rmse']:.4f}")
        print(f"MAE:       {metrics['mae']:.4f}")
        print(f"R² Score:  {metrics['r2']:.4f}")
    
    print("=" * 50)
