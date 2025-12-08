"""
Streamlined Tissue Classification Pipeline
Single config source, TensorBoard integrated, minimal duplication
"""

__version__ = "3.0.0"

# Core - Everything you need
from .config import Config, set_seed, setup_device, create_dirs
from .dataset import TissueDataset, get_transforms, create_dataloaders
from .models import get_model, freeze_backbone, count_parameters
from .trainer import Trainer
from .evaluation import evaluate_model, create_submission
from .visualization import plot_training_history, plot_confusion_matrix

# Optional utilities
from .data_cleaning import create_clean_dataset, analyze_dataset
from .outlier_detection import find_high_loss_samples, visualize_outliers

__all__ = [
    'Config',
    'set_seed', 
    'setup_device',
    'create_dirs',
    'TissueDataset',
    'get_transforms',
    'create_dataloaders',
    'get_model',
    'freeze_backbone',
    'Trainer',
    'evaluate_model',
    'create_submission',
    'plot_training_history',
    'plot_confusion_matrix',
]

