"""
Cross-validation utilities for hyperparameter tuning and model evaluation.

This module provides clean, general-purpose cross-validation functions that work
with any PyTorch model and preprocessed data (numpy arrays).
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from itertools import product
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm

from .trainer import Trainer


def k_fold_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_class: type,
    base_model_params: Dict[str, Any],
    base_trainer_params: Dict[str, Any],
    k: int = 5,
    seed: int = 42,
    use_class_weights: bool = True,
    verbose: bool = True
) -> Tuple[Dict, Dict, Dict]:
    """
    Perform stratified K-fold cross-validation.
    
    This is the RECOMMENDED approach for cross-validation:
    - Each sample appears in exactly one validation fold
    - All data is used for both training and validation
    - Stratification maintains class balance across folds
    - Works with any PyTorch model and preprocessed data
    
    Args:
        X: Feature array of shape (n_samples, seq_length, n_features) or (n_samples, n_features)
        y: Label array of shape (n_samples,)
        model_class: Model class to instantiate (e.g., RecurrentNet)
        base_model_params: Dictionary of model parameters (input_size, hidden_size, etc.)
        base_trainer_params: Dictionary of trainer parameters (epochs, patience, lr, etc.)
        k: Number of folds (default: 5)
        seed: Random seed for reproducibility
        use_class_weights: Whether to compute and use class weights for imbalanced data
        verbose: Whether to print progress
        
    Returns:
        fold_losses: Dictionary of validation loss history per fold
        fold_metrics: Dictionary of validation F1 score history per fold
        best_scores: Dictionary with best F1 per fold plus mean ± std
    """
    device = base_trainer_params.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    batch_size = base_trainer_params.get('batch_size', 32)
    
    # Initialize result containers
    fold_losses = {}
    fold_metrics = {}
    best_scores = {}
    
    # Create stratified K-fold splitter
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    
    # Initialize model to get initial weights
    model = model_class(**base_model_params).to(device)
    initial_state = copy.deepcopy(model.state_dict())
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"K-Fold Cross-Validation (k={k})")
        print(f"{'='*80}")
        print(f"Total samples: {len(X)}")
        print(f"Classes: {np.unique(y)}")
        print(f"Class distribution: {np.bincount(y)}")
    
    # Iterate through folds
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        if verbose:
            print(f"\n{'='*80}")
            print(f"Fold {fold_idx + 1}/{k}")
            print(f"{'='*80}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if verbose:
            print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples")
            print(f"Train class dist: {np.bincount(y_train)}")
            print(f"Val class dist:   {np.bincount(y_val)}")
        
        # Normalize features using training statistics
        # Handle both 2D (n_samples, n_features) and 3D (n_samples, seq_length, n_features)
        if X.ndim == 3:
            # Time series: normalize across both samples and time
            train_min = X_train.min(axis=(0, 1), keepdims=True)
            train_max = X_train.max(axis=(0, 1), keepdims=True)
        else:
            # Regular features: normalize across samples
            train_min = X_train.min(axis=0, keepdims=True)
            train_max = X_train.max(axis=0, keepdims=True)
        
        X_train_norm = (X_train - train_min) / (train_max - train_min + 1e-8)
        X_val_norm = (X_val - train_min) / (train_max - train_min + 1e-8)
        
        # Create datasets
        train_ds = TensorDataset(
            torch.from_numpy(X_train_norm).float(),
            torch.from_numpy(y_train).long()
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val_norm).float(),
            torch.from_numpy(y_val).long()
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        
        # Reset model weights for this fold
        model.load_state_dict(initial_state)
        
        # Compute class weights if requested
        if use_class_weights:
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            criterion = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights).to(device)
            )
            if verbose:
                print(f"Class weights: {class_weights}")
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Create optimizer
        optimizer_type = base_trainer_params.get('optimizer', 'AdamW')
        lr = base_trainer_params.get('learning_rate', 1e-3)
        weight_decay = base_trainer_params.get('weight_decay', 0.0)
        
        if optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Build config dict for Trainer
        cv_config = {
            'training': {
                'epochs': base_trainer_params.get('epochs', 100),
                'patience': base_trainer_params.get('patience', 10),
                'l1_lambda': base_trainer_params.get('l1_lambda', 0.0),
                'l2_lambda': base_trainer_params.get('l2_lambda', 0.0),
                'evaluation_metric': base_trainer_params.get('evaluation_metric', 'val_f1'),
                'mode': base_trainer_params.get('mode', 'max'),
                'restore_best_weights': base_trainer_params.get('restore_best_weights', True),
                'verbose': base_trainer_params.get('verbose', 10),
                'scheduler': base_trainer_params.get('scheduler', {'enabled': False})
            },
            'logging': {
                'save_dir': f'/tmp/cv_fold_{fold_idx}'  # Temporary for CV
            }
        }
        
        # Create trainer (uses existing Trainer class from library)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=cv_config,
            writer=None,  # No TensorBoard for CV
            experiment_name=f"cv_fold_{fold_idx}",
            task='classification'
        )
        
        # Train model
        history = trainer.fit()
        
        # Store results for this fold
        fold_losses[f"fold_{fold_idx}"] = history['val_loss']
        fold_metrics[f"fold_{fold_idx}"] = history['val_f1']
        best_scores[f"fold_{fold_idx}"] = max(history['val_f1'])
        
        if verbose:
            print(f"✓ Best F1 Score: {best_scores[f'fold_{fold_idx}']:.4f}")
    
    # Compute statistics across all folds
    all_best_scores = [best_scores[f"fold_{i}"] for i in range(k)]
    best_scores["mean"] = np.mean(all_best_scores)
    best_scores["std"] = np.std(all_best_scores)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 Cross-Validation Results")
        print(f"{'='*80}")
        print(f"Mean F1: {best_scores['mean']:.4f} ± {best_scores['std']:.4f}")
        print(f"Min F1:  {min(all_best_scores):.4f}")
        print(f"Max F1:  {max(all_best_scores):.4f}")
        print(f"{'='*80}\n")
    
    return fold_losses, fold_metrics, best_scores


def grid_search_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    model_class: type,
    param_grid: Dict[str, List[Any]],
    base_model_params: Dict[str, Any],
    base_trainer_params: Dict[str, Any],
    cv_k: int = 5,
    cv_seed: int = 42,
    use_class_weights: bool = True,
    verbose: bool = True
) -> Tuple[Dict, Dict, float]:
    """
    Execute grid search with K-fold cross-validation for hyperparameter tuning.
    
    Tests all combinations of parameters in param_grid using cross-validation
    to find the best configuration based on mean F1 score.
    
    Args:
        X: Feature array of shape (n_samples, seq_length, n_features) or (n_samples, n_features)
        y: Label array of shape (n_samples,)
        model_class: Model class to instantiate (e.g., RecurrentNet)
        param_grid: Dictionary of parameters to search over
            Model params: hidden_size, num_layers, dropout_rate, rnn_type, bidirectional
            Trainer params: learning_rate, batch_size, weight_decay
        base_model_params: Base model parameters (fixed, not searched)
        base_trainer_params: Base trainer parameters (fixed, not searched)
        cv_k: Number of CV folds per configuration (default: 5)
        cv_seed: Random seed for CV
        use_class_weights: Whether to use class weights
        verbose: Whether to print progress
        
    Returns:
        all_results: Dictionary with detailed results for each configuration
        best_params: Dictionary with best hyperparameter combination
        best_score: Best mean F1 score achieved
    """
    # Define which parameters belong to model vs trainer
    model_param_names = {
        'input_size', 'hidden_size', 'num_layers', 'num_classes',
        'rnn_type', 'bidirectional', 'dropout_rate', 'task'
    }
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    all_results = {}
    best_score = -np.inf
    best_params = None
    
    total = len(combinations)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Grid Search Hyperparameter Tuning")
        print(f"{'='*80}")
        print(f"Total configurations: {total}")
        print(f"CV folds per config: {cv_k}")
        print(f"Total training runs: {total * cv_k}")
        print(f"{'='*80}\n")
    
    # Iterate through all parameter combinations
    for idx, combo in enumerate(combinations, 1):
        # Create current configuration
        current_config = dict(zip(param_names, combo))
        config_str = "_".join([f"{k}={v}" for k, v in current_config.items()])
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Configuration {idx}/{total}")
            print(f"{'='*80}")
            for param, value in current_config.items():
                print(f"  {param}: {value}")
        
        # Separate model and trainer parameters
        model_params = base_model_params.copy()
        trainer_params = base_trainer_params.copy()
        
        for key, value in current_config.items():
            if key in model_param_names:
                model_params[key] = value
            else:
                trainer_params[key] = value
        
        # Run cross-validation for this configuration
        try:
            _, _, fold_scores = k_fold_cross_validation(
                X=X,
                y=y,
                model_class=model_class,
                base_model_params=model_params,
                base_trainer_params=trainer_params,
                k=cv_k,
                seed=cv_seed,
                use_class_weights=use_class_weights,
                verbose=False  # Suppress individual fold output
            )
            
            # Store detailed results
            all_results[config_str] = {
                'params': current_config.copy(),
                'mean_f1': fold_scores['mean'],
                'std_f1': fold_scores['std'],
                'fold_scores': fold_scores
            }
            
            # Track best configuration
            if fold_scores['mean'] > best_score:
                best_score = fold_scores['mean']
                best_params = current_config.copy()
                if verbose:
                    print(f"\n  ⭐ NEW BEST CONFIGURATION!")
            
            if verbose:
                print(f"\n  Results:")
                print(f"    Mean F1: {fold_scores['mean']:.4f} ± {fold_scores['std']:.4f}")
                print(f"    Min F1:  {min([fold_scores[f'fold_{i}'] for i in range(cv_k)]):.4f}")
                print(f"    Max F1:  {max([fold_scores[f'fold_{i}'] for i in range(cv_k)]):.4f}")
        
        except Exception as e:
            print(f"\n  ❌ Configuration failed: {e}")
            all_results[config_str] = {
                'params': current_config.copy(),
                'mean_f1': 0.0,
                'std_f1': 0.0,
                'error': str(e)
            }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🏆 Grid Search Complete!")
        print(f"{'='*80}")
        print(f"Best Configuration:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\nBest Mean F1 Score: {best_score:.4f}")
        print(f"{'='*80}\n")
    
    return all_results, best_params, best_score


def plot_cv_results(
    fold_metrics: Dict,
    k: int,
    metric_name: str = 'F1 Score',
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Visualize cross-validation results with training curves.
    
    Args:
        fold_metrics: Dictionary from k_fold_cross_validation with metric history per fold
        k: Number of folds
        metric_name: Name of the metric to display (default: 'F1 Score')
        figsize: Figure size (default: (12, 6))
        
    Example:
        >>> fold_losses, fold_metrics, best_scores = k_fold_cross_validation(...)
        >>> plot_cv_results(fold_metrics, k=5)
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Training curves for all folds
    for i in range(k):
        fold_key = f"fold_{i}"
        if fold_key in fold_metrics:
            scores = fold_metrics[fold_key]
            axes[0].plot(scores, label=f'Fold {i+1}', alpha=0.7)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel(metric_name)
    axes[0].set_title(f'{metric_name} Across Folds')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Boxplot of best scores per fold
    best_scores = [max(fold_metrics[f"fold_{i}"]) for i in range(k)]
    axes[1].boxplot([best_scores], labels=['All Folds'])
    axes[1].scatter([1] * len(best_scores), best_scores, alpha=0.5, s=100)
    axes[1].set_ylabel(f'Best {metric_name}')
    axes[1].set_title(f'Distribution of Best {metric_name}')
    axes[1].grid(alpha=0.3, axis='y')
    
    # Add mean and std annotations
    mean_score = np.mean(best_scores)
    std_score = np.std(best_scores)
    axes[1].axhline(mean_score, color='r', linestyle='--', alpha=0.7,
                    label=f'Mean: {mean_score:.4f}±{std_score:.4f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_grid_search_results(
    all_results: Dict,
    top_n: int = 10,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Visualize grid search results showing top N configurations.
    
    Args:
        all_results: Results dictionary from grid_search_hyperparameters
        top_n: Number of top configurations to display (default: 10)
        figsize: Figure size (default: (14, 8))
        
    Example:
        >>> all_results, best_params, best_score = grid_search_hyperparameters(...)
        >>> plot_grid_search_results(all_results, top_n=10)
    """
    import matplotlib.pyplot as plt
    
    # Sort by mean F1 score
    sorted_configs = sorted(
        [(name, data) for name, data in all_results.items() if 'error' not in data],
        key=lambda x: x[1]['mean_f1'],
        reverse=True
    )[:top_n]
    
    if not sorted_configs:
        print("⚠️  No valid results to plot.")
        return None
    
    # Extract data
    config_names = []
    mean_scores = []
    std_scores = []
    
    for name, data in sorted_configs:
        # Create short label
        params = data['params']
        label_parts = []
        for k, v in params.items():
            if k == 'hidden_size':
                label_parts.append(f"H={v}")
            elif k == 'num_layers':
                label_parts.append(f"L={v}")
            elif k == 'learning_rate':
                label_parts.append(f"LR={v}")
            elif k == 'dropout_rate':
                label_parts.append(f"D={v}")
            elif k == 'rnn_type':
                label_parts.append(f"{v}")
            elif k == 'batch_size':
                label_parts.append(f"BS={v}")
        
        config_names.append("\n".join(label_parts) if len(label_parts) <= 3 else ", ".join(label_parts))
        mean_scores.append(data['mean_f1'])
        std_scores.append(data['std_f1'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(config_names))
    bars = ax.bar(x_pos, mean_scores, yerr=std_scores, capsize=5, alpha=0.7,
                  color=['green' if i == 0 else 'lightblue' for i in range(len(config_names))])
    
    # Highlight best configuration
    bars[0].set_edgecolor('darkgreen')
    bars[0].set_linewidth(2)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Mean F1 Score', fontsize=12)
    ax.set_title(f'Top {len(sorted_configs)} Hyperparameter Configurations', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(mean_scores, std_scores)):
        ax.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}',
               ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return fig