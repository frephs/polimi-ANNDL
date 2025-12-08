"""
Single source of truth for all configuration
ALL VALUES MUST COME FROM YAML - no hardcoded defaults
"""
import os
import random
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")


class Config:
    """Configuration loaded from YAML file only - no defaults in code"""
    
    def __init__(self, yaml_path='config.yaml'):
        """Load all configuration from YAML file
        
        Args:
            yaml_path: Path to YAML config file (required)
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required. Install: pip install pyyaml")
        
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {yaml_path}\n"
                f"Please create config.yaml or specify path to existing config."
            )
        
        # Load YAML
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if not config_dict:
            raise ValueError(f"Empty config file: {yaml_path}")
        
        # Set all attributes from YAML
        for key, value in config_dict.items():
            setattr(self, key, value)
        
        # Validate required keys
        self._validate()
        
        print(f"✅ Config loaded from {yaml_path}")
        print(f"   {len(config_dict)} parameters loaded")
    
    def _validate(self):
        """Validate that required config keys are present"""
        required = [
            # Reproducibility
            'SEED', 'EXPERIMENT_NAME',
            # Paths
            'DATA_DIR', 'CSV_PATH', 'TEST_DIR', 'OUTPUT_DIR', 'MODELS_DIR',
            'CHECKPOINT_DIR', 'TENSORBOARD_DIR', 'VISUALIZATION_DIR',
            # Data
            'IMG_SIZE', 'VAL_SPLIT', 'BATCH_SIZE', 'NUM_WORKERS', 'PIN_MEMORY', 'PREFETCH_FACTOR',
            # Model (NUM_CLASSES is auto-detected, not required in YAML)
            'MODEL_NAME', 'USE_PRETRAINED', 'FREEZE_BACKBONE', 'USE_ATTENTION', 'DROPOUT',
            # Training
            'NUM_EPOCHS', 'LEARNING_RATE', 'WEIGHT_DECAY', 'OPTIMIZER',
            # Loss function
            'USE_WEIGHTED_LOSS', 'CLASS_WEIGHTS', 'USE_FOCAL_LOSS', 'FOCAL_ALPHA', 'FOCAL_GAMMA',
            # Scheduler
            'USE_SCHEDULER', 'SCHEDULER', 'PATIENCE_SCHEDULER', 'FACTOR',
            # Early stopping & checkpointing
            'PATIENCE', 'MONITOR_METRIC', 'CHECKPOINT_INTERVAL', 'SAVE_BEST_ONLY',
            # Advanced training
            'USE_MIXED_PRECISION', 'GRADIENT_CLIP', 'LABEL_SMOOTHING',
            'USE_PROGRESSIVE_TRAINING', 'FREEZE_EPOCHS', 'UNFREEZE_LR_MULTIPLIER',
            # Augmentation
            'USE_AUGMENTATION', 'USE_AUTOMATED_AUG', 'AUTOMATED_AUG_METHOD',
            'RANDAUGMENT_N', 'RANDAUGMENT_M',
            'HORIZONTAL_FLIP', 'VERTICAL_FLIP', 'ROTATION_DEGREES', 'COLOR_JITTER', 'RANDOM_ERASING',
            'USE_MIXUP', 'MIXUP_ALPHA', 'MIXUP_PROB', 
            'USE_CUTMIX', 'CUTMIX_ALPHA', 'CUTMIX_PROB',
            'USE_TTA', 'TTA_NUM_AUGMENTATIONS',
            # Normalization
            'USE_IMAGENET_NORM', 'NORM_TYPE',
            # Visualization
            'VISUALIZE_ATTENTION', 'VISUALIZE_ACTIVATIONS', 'SAVE_ATTENTION_MAPS',
            # TensorBoard
            'USE_TENSORBOARD', 'LOG_IMAGES', 'LOG_HISTOGRAMS', 'LOG_INTERVAL'
        ]
        
        missing = [k for k in required if not hasattr(self, k)]
        if missing:
            raise ValueError(
                f"Missing required config keys: {missing}\n"
                f"Please add them to your config.yaml\n"
                f"See config.yaml for complete template with all required fields."
            )
    
    @classmethod
    def from_yaml(cls, yaml_path):
        """Load config from YAML file (alias for __init__)"""
        return cls(yaml_path)
    
    def save_yaml(self, yaml_path):
        """Save current config to YAML file"""
        config_dict = {k: v for k, v in vars(self).items() 
                      if not k.startswith('_')}
        
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Config saved to {yaml_path}")
    
    def __repr__(self):
        """Pretty print config"""
        items = [f"{k}={v}" for k, v in vars(self).items() 
                if not k.startswith('_')]
        return f"Config({len(items)} params)"


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_device():
    """Setup and return the best available device"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            print("⚠️  Using CPU (training will be slow)")
        return device
    except ImportError:
        print("⚠️  PyTorch not installed, assuming CPU")
        return None


def create_dirs(config):
    """Create necessary directories from config"""
    dirs = []
    for attr in ['OUTPUT_DIR', 'MODELS_DIR', 'CHECKPOINT_DIR', 
                 'TENSORBOARD_DIR', 'VISUALIZATION_DIR']:
        if hasattr(config, attr):
            dirs.append(getattr(config, attr))
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

