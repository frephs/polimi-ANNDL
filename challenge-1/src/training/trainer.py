"""
Training loop and utilities
"""

import os
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, Literal
from tqdm import tqdm

from .metrics import calculate_metrics


class Trainer:
    """
    Trainer class for training and evaluating recurrent models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict,
        writer: Optional[SummaryWriter] = None,
        experiment_name: str = "experiment",
        task: Literal['classification', 'regression'] = 'classification'
    ):
        """
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device (cuda/cpu)
            config: Configuration dictionary
            writer: TensorBoard writer
            experiment_name: Name for saving models
            task: 'classification' or 'regression' - determines metrics and logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.writer = writer
        self.experiment_name = experiment_name
        self.task = task
        
        # Training timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Training configuration
        training_config = config['training']
        self.epochs = training_config['epochs']
        self.patience = training_config['patience']
        self.l1_lambda = training_config['l1_lambda']
        self.l2_lambda = training_config['l2_lambda']
        self.evaluation_metric = training_config['evaluation_metric']
        self.mode = training_config['mode']
        self.restore_best_weights = training_config['restore_best_weights']
        self.verbose = training_config['verbose']
        
        # Mixed precision scaler (only enable for CUDA)
        self.use_amp = (device.type == 'cuda')
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        
        # Create save directory
        self.save_dir = config['logging']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize scheduler if configured
        self.scheduler = self._create_scheduler()
        
        # Training history (task-specific metrics)
        if self.task == 'classification':
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'train_f1': [],
                'val_f1': []
            }
            self.primary_metric = 'f1'
        else:  # regression
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'train_r2': [],
                'val_r2': []
            }
            self.primary_metric = 'r2'
        
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        scheduler_config = self.config['training']['scheduler']
        
        if not scheduler_config['enabled']:
            return None
        
        if scheduler_config['type'] == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min' if self.mode == 'min' else 'max',
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience'],
                min_lr=scheduler_config['min_lr']
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
    
    def train_one_epoch(self) -> Tuple[float, float]:
        """
        Perform one training epoch.
        
        Returns:
            Tuple of (average_loss, primary_metric_value)
        """
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision (only on CUDA)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Add L1 and L2 regularization
                if self.l1_lambda > 0:
                    l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                    loss = loss + self.l1_lambda * l1_norm
                
                if self.l2_lambda > 0:
                    l2_norm = sum(p.pow(2).sum() for p in self.model.parameters())
                    loss = loss + self.l2_lambda * l2_norm
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Accumulate metrics
            running_loss += loss.item() * inputs.size(0)
            
            if self.task == 'classification':
                predictions = outputs.argmax(dim=1)
            else:  # regression
                predictions = outputs
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader.dataset)
        metrics = calculate_metrics(
            np.concatenate(all_targets),
            np.concatenate(all_predictions),
            task=self.task
        )
        
        return epoch_loss, metrics[self.primary_metric]
    
    def validate_one_epoch(self) -> Tuple[float, float]:
        """
        Perform one validation epoch.
        
        Returns:
            Tuple of (average_loss, primary_metric_value)
        """
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                
                if self.task == 'classification':
                    predictions = outputs.argmax(dim=1)
                else:  # regression
                    predictions = outputs
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        metrics = calculate_metrics(
            np.concatenate(all_targets),
            np.concatenate(all_predictions),
            task=self.task
        )
        
        return epoch_loss, metrics[self.primary_metric]
    
    def log_to_tensorboard(self, epoch: int, train_loss: float, train_metric: float,
                          val_loss: float, val_metric: float) -> None:
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return
        
        # Log losses
        self.writer.add_scalar('Loss/Training', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        # Log primary metric (F1 for classification, R² for regression)
        metric_name = 'F1' if self.task == 'classification' else 'R2'
        self.writer.add_scalar(f'{metric_name}/Training', train_metric, epoch)
        self.writer.add_scalar(f'{metric_name}/Validation', val_metric, epoch)
        
        # Log learning rate
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'LearningRate/group_{i}', param_group['lr'], epoch)
        
        # Log model parameters and gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.numel() > 0:
                self.writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                if param.grad is not None and param.grad.numel() > 0:
                    if torch.isfinite(param.grad).all():
                        self.writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)
    
    def fit(self) -> Dict:
        """
        Train the model for the specified number of epochs.
        Saves models with timestamp and best metric value when performance improves.
        
        Returns:
            Training history dictionary
        """
        best_metric = float('-inf') if self.mode == 'max' else float('inf')
        best_epoch = 0
        patience_counter = 0
        
        # Create base model path with timestamp
        base_model_name = f"{self.experiment_name}_{self.timestamp}"
        
        print(f"Training for {self.epochs} epochs...")
        print(f"Task: {self.task.upper()} | Primary metric: {self.primary_metric.upper()}")
        print(f"Device: {self.device} | Mixed Precision: {'Enabled' if self.use_amp else 'Disabled (CPU only)'}")
        print(f"Logging to TensorBoard: {self.writer is not None}")
        print(f"Models will be saved to: {self.save_dir}")
        print("-" * 80)
        
        for epoch in range(1, self.epochs + 1):
            # Training
            train_loss, train_metric = self.train_one_epoch()
            
            # Validation
            val_loss, val_metric = self.validate_one_epoch()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if self.mode == 'min' else val_metric)
                else:
                    self.scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            if self.task == 'classification':
                self.history['train_f1'].append(train_metric)
                self.history['val_f1'].append(val_metric)
            else:
                self.history['train_r2'].append(train_metric)
                self.history['val_r2'].append(val_metric)
            
            # Log to TensorBoard
            self.log_to_tensorboard(epoch, train_loss, train_metric, val_loss, val_metric)
            
            # Print progress
            metric_display = 'F1' if self.task == 'classification' else 'R²'
            if self.verbose > 0 and (epoch % self.verbose == 0 or epoch == 1):
                print(f"Epoch {epoch:3d}/{self.epochs} | "
                      f"Train: Loss={train_loss:.4f}, {metric_display}={train_metric:.4f} | "
                      f"Val: Loss={val_loss:.4f}, {metric_display}={val_metric:.4f}")
            
            # Early stopping logic and model saving
            if self.patience > 0:
                current_metric = self.history[self.evaluation_metric][-1]
                is_improvement = (
                    (current_metric > best_metric) if self.mode == 'max'
                    else (current_metric < best_metric)
                )
                
                if is_improvement:
                    best_metric = current_metric
                    best_epoch = epoch
                    
                    # Save model with timestamp and best metric
                    metric_str = f"{self.primary_metric}{best_metric:.4f}".replace('.', '_')
                    model_path = os.path.join(
                        self.save_dir, 
                        f"{base_model_name}_epoch{epoch:03d}_best_{metric_str}.pt"
                    )
                    
                    # Save model state dict and metadata
                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_metric': best_metric,
                        'metric_name': self.primary_metric,
                        'task': self.task,
                        'timestamp': self.timestamp,
                        'config': self.config
                    }
                    torch.save(save_dict, model_path)
                    
                    if self.verbose > 0:
                        print(f"✓ Model improved! Saved to: {os.path.basename(model_path)}")
                    
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"\nEarly stopping triggered after {epoch} epochs.")
                        print(f"Best {self.primary_metric.upper()}: {best_metric:.4f} at epoch {best_epoch}")
                        break
        
        # Load best model weights
        if self.restore_best_weights and self.patience > 0 and best_epoch > 0:
            # Find the best model file
            metric_str = f"{self.primary_metric}{best_metric:.4f}".replace('.', '_')
            model_path = os.path.join(
                self.save_dir,
                f"{base_model_name}_epoch{best_epoch:03d}_best_{metric_str}.pt"
            )
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"\n✓ Best model restored from epoch {best_epoch}")
                print(f"  {self.primary_metric.upper()}: {best_metric:.4f}")
        
        # Save final model if no early stopping
        if self.patience == 0:
            final_model_path = os.path.join(
                self.save_dir,
                f"{base_model_name}_final.pt"
            )
            save_dict = {
                'epoch': self.epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'task': self.task,
                'timestamp': self.timestamp,
                'config': self.config
            }
            torch.save(save_dict, final_model_path)
            print(f"\n✓ Final model saved to: {os.path.basename(final_model_path)}")
        
        if self.writer is not None:
            self.writer.close()
            print(f"✓ TensorBoard logs saved")
        
        print("-" * 80)
        print(f"Training complete!")
        
        return self.history
