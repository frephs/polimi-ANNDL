"""
Streamlined trainer with TensorBoard, mixed precision, and checkpointing
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Trainer:
    """Production-ready trainer with all features integrated"""
    
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        label_smoothing = getattr(config, 'LABEL_SMOOTHING')
        use_focal_loss = getattr(config, 'USE_FOCAL_LOSS')
        use_weighted_loss = getattr(config, 'USE_WEIGHTED_LOSS')
        
        if use_focal_loss:
            focal_alpha = getattr(config, 'FOCAL_ALPHA')
            focal_gamma = getattr(config, 'FOCAL_GAMMA')
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print(f"📊 Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        elif use_weighted_loss:
            # Class weights: inverse frequency normalized
            # Triple neg: 156, Luminal A: 414, Luminal B: 445, HER2+: 397
            class_weights = getattr(config, 'CLASS_WEIGHTS')
            weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
            self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
            print(f"⚖️  Using Weighted Loss: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Optimizer
        optimizer_name = config.OPTIMIZER.lower()
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        elif optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=config.LEARNING_RATE, 
                weight_decay=config.WEIGHT_DECAY, momentum=0.9, nesterov=True)
        elif optimizer_name == 'lion':
            # Lion imported at module level
            self.optimizer = Lion(
                model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        elif optimizer_name == 'ranger':
            # Ranger imported at module level
            self.optimizer = Ranger(
                model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        else:  # adamw (default)
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        
        # Scheduler
        self.scheduler = None
        if getattr(config, 'USE_SCHEDULER'):
            scheduler_type = getattr(config, 'SCHEDULER').lower()
            if scheduler_type == 'plateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='max', 
                    factor=getattr(config, 'FACTOR'),
                    patience=getattr(config, 'PATIENCE_SCHEDULER'),
                    min_lr=1e-6)
            elif scheduler_type == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
            elif scheduler_type == 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=15, gamma=0.1)
        
        # Mixed precision
        self.use_amp = getattr(config, 'USE_MIXED_PRECISION') and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.gradient_clip = getattr(config, 'GRADIENT_CLIP')
        
        # TensorBoard
        self.writer = None
        if getattr(config, 'USE_TENSORBOARD'):
            log_dir = f"{config.TENSORBOARD_DIR}/{time.strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir)
            print(f"✅ TensorBoard: {log_dir}")
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.history = []
        self.patience_counter = 0
        
        # CutMix/MixUp augmentation
        self.use_cutmix = getattr(config, 'USE_CUTMIX')
        self.use_mixup = getattr(config, 'USE_MIXUP')
        if self.use_cutmix or self.use_mixup:
            # CutMix/MixUp imported at module level
            self.mixup_fn = None
            if self.use_cutmix:
                cutmix_alpha = getattr(config, 'CUTMIX_ALPHA')
                cutmix_prob = getattr(config, 'CUTMIX_PROB')
                self.mixup_fn = CutMix(alpha=cutmix_alpha, prob=cutmix_prob)
                print(f"✂️  Using CutMix (alpha={cutmix_alpha}, prob={cutmix_prob})")
            elif self.use_mixup:
                mixup_alpha = getattr(config, 'MIXUP_ALPHA')
                mixup_prob = getattr(config, 'MIXUP_PROB')
                self.mixup_fn = MixUp(alpha=mixup_alpha, prob=mixup_prob)
                print(f"🔀 Using MixUp (alpha={mixup_alpha}, prob={mixup_prob})")
            self.mixup_criterion = mixup_criterion
        else:
            self.mixup_fn = None
        
        print(f"✅ Trainer ready - {optimizer_name.upper()}, AMP={self.use_amp}")
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Apply CutMix/MixUp if enabled
            if self.mixup_fn is not None:
                images, labels_a, labels_b, lam = self.mixup_fn(images, labels)
                use_mixup = True
            else:
                labels_a = labels
                labels_b = None
                lam = 1.0
                use_mixup = False
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward with AMP
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                if use_mixup:
                    loss = self.mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
            
            # Metrics (use original labels for tracking)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_a.cpu().numpy())
            
            # TensorBoard batch logging
            if self.writer and batch_idx % getattr(self.config, 'LOG_INTERVAL') == 0:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Batch/Loss', loss.item(), step)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc, epoch_f1
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return val_loss, val_acc, val_f1
    
    def save_checkpoint(self, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'history': self.history
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest
        latest_path = os.path.join(self.config.CHECKPOINT_DIR, 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            exp_name = getattr(self.config, 'EXPERIMENT_NAME')
            best_path = os.path.join(self.config.MODELS_DIR, f'{exp_name}_best.pt')
            torch.save(self.model.state_dict(), best_path)
            print(f"💾 Best: {self.best_metric:.4f} -> {best_path}")
    
    def freeze_backbone(self):
        """Freeze backbone layers for progressive training"""
        frozen_count = 0
        for name, param in self.model.named_parameters():
            # Freeze all except classifier/fc layer
            if 'classifier' not in name and 'fc' not in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"❄️  Froze {frozen_count} backbone parameters")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        unfrozen_count = 0
        for param in self.model.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1
        print(f"🔥 Unfroze {unfrozen_count} parameters for fine-tuning")
    
    def load_checkpoint(self, path=None):
        """Load checkpoint"""
        if path is None:
            path = os.path.join(self.config.CHECKPOINT_DIR, 'latest.pt')
        
        if not os.path.exists(path):
            return False
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint['best_metric']
        self.history = checkpoint['history']
        
        print(f"✅ Resumed from epoch {self.current_epoch}")
        return True
    
    def train(self, resume=False):
        """Main training loop"""
        if resume:
            self.load_checkpoint()
        
        print(f"\n🚀 Training: {self.config.NUM_EPOCHS} epochs")
        print(f"   Device: {self.device} | Batch: {self.config.BATCH_SIZE} | LR: {self.config.LEARNING_RATE}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # Train & validate
            train_loss, train_acc, train_f1 = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()
            
            # Track history
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1,
                'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # TensorBoard
            if self.writer:
                self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
                self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
                self.writer.add_scalars('F1', {'train': train_f1, 'val': val_f1}, epoch)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
                
                # Log model weights
                if getattr(self.config, 'LOG_HISTOGRAMS'):
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            self.writer.add_histogram(f'Weights/{name}', param.data, epoch)
                            if param.grad is not None:
                                self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            
            # Print
            print(f"\n[{epoch+1}/{self.config.NUM_EPOCHS}]")
            print(f"  Train: Loss={train_loss:.4f} Acc={train_acc:.4f} F1={train_f1:.4f}")
            print(f"  Val:   Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}")
            
            # Check improvement
            monitor_metric = getattr(self.config, 'MONITOR_METRIC')
            current_metric = val_f1 if 'f1' in monitor_metric else (val_acc if 'acc' in monitor_metric else -val_loss)
            
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save
            if getattr(self.config, 'SAVE_BEST_ONLY'):
                if is_best:
                    self.save_checkpoint(is_best=True)
            else:
                self.save_checkpoint(is_best=is_best)
                if (epoch + 1) % self.config.CHECKPOINT_INTERVAL == 0:
                    epoch_path = os.path.join(self.config.CHECKPOINT_DIR, f'epoch_{epoch+1}.pt')
                    torch.save(self.model.state_dict(), epoch_path)
            
            # Early stopping
            if self.patience_counter >= self.config.PATIENCE:
                print(f"\n⏹️  Early stop ({self.config.PATIENCE} epochs no improvement)")
                break
        
        elapsed = (time.time() - start_time) / 60
        print(f"\n✅ Done in {elapsed:.1f}min | Best: {self.best_metric:.4f}")
        
        if self.writer:
            self.writer.close()
        
        return self.history
    
    def get_history_df(self):
        """Get training history as pandas DataFrame"""
        return pd.DataFrame(self.history)
