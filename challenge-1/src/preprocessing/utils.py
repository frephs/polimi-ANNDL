"""
Data preprocessing utilities for time series classification
"""
import random
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from datetime import datetime




def preprocess_pirates_data(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    config: Dict,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess pirates pain classification data according to config.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training labels DataFrame
        config: Configuration dictionary
        verbose: Print preprocessing steps
        
    Returns:
        Preprocessed X_train and y_train DataFrames
    """
    if verbose:
        print("=" * 80)
        print("PREPROCESSING PIRATES PAIN DATA")
        print("=" * 80)
    
    if config.get('time_features', {}).get('enabled', False):
        X_train = extract_time_features(
            X_train,
            time_column='time',
            extract_hour=config['time_features'].get('extract_hour', True),
            extract_day_of_week=config['time_features'].get('extract_day_of_week', True),
            extract_day_of_month=config['time_features'].get('extract_day_of_month', False),
            use_cyclical_encoding=config['time_features'].get('use_cyclical_encoding', True),
            verbose=verbose
        )
    
    # Drop constant/redundant features
    for col in config['preprocessing']['drop_features']:
        if col in X_train.columns:
            X_train = X_train.drop(col, axis=1)
            if verbose:
                print(f"Dropped feature: {col}")
    
    # Combine highly correlated features
    if config['preprocessing']['combine_correlations']:
        # Combine joint_00 and joint_02
        if 'joint_00' in X_train.columns and 'joint_02' in X_train.columns:
            X_train['joint_00_02'] = (X_train['joint_00'] + X_train['joint_02']) / 2
            X_train = X_train.drop(["joint_00", "joint_02"], axis=1)
            if verbose:
                print("Combined joint_00 and joint_02 → joint_00_02")
        
        # Combine joint_01 and joint_03
        if 'joint_01' in X_train.columns and 'joint_03' in X_train.columns:
            X_train['joint_01_03'] = (X_train['joint_01'] + X_train['joint_03']) / 2
            X_train = X_train.drop(["joint_01", "joint_03"], axis=1)
            if verbose:
                print("Combined joint_01 and joint_03 → joint_01_03")
        
        # Combine joint_10 and joint_11
        if 'joint_10' in X_train.columns and 'joint_11' in X_train.columns:
            X_train['joint_10_11'] = (X_train['joint_10'] + X_train['joint_11']) / 2
            X_train = X_train.drop(["joint_10", "joint_11"], axis=1)
            if verbose:
                print("Combined joint_10 and joint_11 → joint_10_11")

    # Create prosthesis indicator feature
    if all(col in X_train.columns for col in ['n_legs', 'n_hands', 'n_eyes']):
        X_train["has_peg_leg"] = (X_train["n_legs"] == "one+peg_leg").astype(int)
        X_train["has_hook_hand"] = (X_train["n_hands"] == "one+hook_hand").astype(int)
        X_train["has_eye_patch"] = (X_train["n_eyes"] == "one+eye_patch").astype(int)
        X_train["has_prosthesis"] = X_train[["has_peg_leg", "has_hook_hand", "has_eye_patch"]].max(axis=1).astype(int)
        
        # Drop individual columns
        X_train = X_train.drop(["n_legs", "n_hands", "n_eyes", "has_peg_leg", "has_hook_hand", "has_eye_patch"], axis=1)
        if verbose:
            print("Created has_prosthesis feature from body part indicators")

    # Map labels to integers
    labels_map = config['labels']
    y_train['label'] = y_train['label'].map(labels_map)
    if verbose:
        print(f"Mapped labels: {labels_map}")
    
    if verbose:
        print(f"\nFinal shape: {X_train.shape}")
        print("=" * 80)
    
    return X_train, y_train


def split_train_val(
    X: pd.DataFrame,
    y: pd.DataFrame,
    val_size: float = 0.15,
    stratify: bool = True,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and validation sets based on unique users with stratification.
    
    Args:
        X: Features DataFrame (must have 'sample_index' column)
        y: Labels DataFrame (must have 'sample_index' and 'label' columns)
        val_size: Fraction of users to use for validation (0.0 to 1.0)
        stratify: If True, maintain class balance across train/val splits
        seed: Random seed for reproducibility
        
    Returns:
        X_train, X_val, y_train, y_val
    """
    from sklearn.model_selection import train_test_split
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Get unique user IDs
    unique_users = X['sample_index'].unique()
    
    if stratify:
        # For each user, get their majority label (most common label for that user)
        user_labels = []
        for user_id in unique_users:
            user_data = y[y['sample_index'] == user_id]
            # Get most common label for this user
            majority_label = user_data['label'].mode()[0]
            user_labels.append(majority_label)
        
        user_labels = np.array(user_labels)
        
        # Stratified split to maintain class balance
        train_users, val_users = train_test_split(
            unique_users,
            test_size=val_size,
            stratify=user_labels,
            random_state=seed
        )
        
        print(f"Stratified split by user labels:")
        print(f"  Train users: {len(train_users)} ({(1-val_size)*100:.1f}%)")
        print(f"  Val users:   {len(val_users)} ({val_size*100:.1f}%)")
        
    else:
        # Random split without stratification
        n_val_users = int(len(unique_users) * val_size)
        shuffled_users = np.random.permutation(unique_users)
        train_users = shuffled_users[:-n_val_users]
        val_users = shuffled_users[-n_val_users:]
        
        print(f"Random split:")
        print(f"  Train users: {len(train_users)} ({(1-val_size)*100:.1f}%)")
        print(f"  Val users:   {len(val_users)} ({val_size*100:.1f}%)")
    
    # Split data based on user IDs
    X_train = X[X['sample_index'].isin(train_users)].copy()
    X_val = X[X['sample_index'].isin(val_users)].copy()
    y_train = y[y['sample_index'].isin(train_users)].copy()
    y_val = y[y['sample_index'].isin(val_users)].copy()
    
    # Print class distribution in both sets
    print(f"\nTrain: {X_train.shape[0]} samples from {len(train_users)} users")
    train_class_dist = y_train['label'].value_counts(normalize=True).sort_index()
    for cls, pct in train_class_dist.items():
        print(f"  Class {cls}: {pct*100:5.2f}%")
    
    print(f"\nVal:   {X_val.shape[0]} samples from {len(val_users)} users")
    val_class_dist = y_val['label'].value_counts(normalize=True).sort_index()
    for cls, pct in val_class_dist.items():
        print(f"  Class {cls}: {pct*100:5.2f}%")
    
    return X_train, X_val, y_train, y_val


def normalize_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    feature_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize features using min-max scaling fitted ONLY on training set.
    
    Args:
        X_train: Training features
        X_val: Validation features
        feature_columns: List of columns to normalize
        
    Returns:
        Normalized X_train and X_val (both normalized using training statistics)
    """
    # FIXED: Calculate min and max ONLY from training set (prevents data leakage)
    train_mins = X_train[feature_columns].min()
    train_maxs = X_train[feature_columns].max()

    # Apply normalization using TRAINING statistics to both sets
    X_train = X_train.copy()
    X_val = X_val.copy()
    
    for column in feature_columns:
        # Avoid division by zero for constant features
        denominator = train_maxs[column] - train_mins[column]
        if denominator == 0:
            denominator = 1.0
        
        X_train[column] = (X_train[column] - train_mins[column]) / denominator
        X_val[column] = (X_val[column] - train_mins[column]) / denominator

    return X_train, X_val


def build_sequences(
    df: pd.DataFrame,
    window: int = 50,
    stride: int = 25,
    feature_columns: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sequences from time series data with sliding windows.
    
    Args:
        df: DataFrame with columns ['sample_index', 'label', ...features...]
        window: Size of the sliding window (default: 50, must be <= 160 for this dataset)
        stride: Step size for the sliding window (default: 25, creates 50% overlap)
        feature_columns: List of feature column names to use
        
    Returns:
        dataset: Array of sequences with shape (n_samples, window, n_features)
        labels: Array of labels with shape (n_samples,)
    """
    # If feature_columns not provided, infer from dataframe
    if feature_columns is None:
        # Exclude metadata columns
        exclude_cols = ['sample_index', 'time', 'label']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    # Initialize lists
    dataset = []
    labels = []
    
    # Iterate over unique sample IDs
    for sample_id in df['sample_index'].unique():
        # Extract data for current sample
        sample_data = df[df['sample_index'] == sample_id][feature_columns].values.astype(np.float32)
        
        # Get label
        label = df[df['sample_index'] == sample_id]['label'].values[0]
        
        # Data quality check: validate sufficient timesteps for windowing
        if len(sample_data) < window:
            print(f"WARNING: Sample {sample_id} has {len(sample_data)} timesteps but window={window}.")
            print(f"         Skipping this sample. Consider using a smaller window size.")
            continue  # Skip samples with insufficient data
        
        # Build windows with sliding window
        idx = 0
        num_windows = 0
        while idx + window <= len(sample_data):
            dataset.append(sample_data[idx:idx + window])
            labels.append(label)
            idx += stride
            num_windows += 1
    
    # Convert to arrays
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    print(f"Built {len(dataset)} sequences from {len(df['sample_index'].unique())} users")
    print(f"Sequence shape: {dataset.shape}")
    
    return dataset, labels


def augment_time_series(
    sequence: np.ndarray,
    noise_level: float = 0.01,
    scale_range: Tuple[float, float] = (0.95, 1.05),
    shift_range: int = 5,
    apply_noise: bool = True,
    apply_scaling: bool = True,
    apply_shift: bool = True
) -> np.ndarray:
    """
    Apply data augmentation to a time series sequence.
    
    Args:
        sequence: Time series sequence with shape (window, features)
        noise_level: Standard deviation of Gaussian noise to add
        scale_range: Range for random scaling factor
        shift_range: Maximum number of timesteps to shift
        apply_noise: Whether to add Gaussian noise
        apply_scaling: Whether to apply random scaling
        apply_shift: Whether to apply time shifting
        
    Returns:
        Augmented sequence with same shape
    """
    augmented = sequence.copy()
    
    # 1. Add Gaussian noise
    if apply_noise:
        noise = np.random.normal(0, noise_level, sequence.shape)
        augmented = augmented + noise
    
    # 2. Random scaling
    if apply_scaling:
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        augmented = augmented * scale_factor
    
    # 3. Time shifting (roll)
    if apply_shift:
        shift = np.random.randint(-shift_range, shift_range + 1)
        augmented = np.roll(augmented, shift, axis=0)
    
    return augmented.astype(np.float32)


def oversample_minority_classes(
    X_sequences: np.ndarray,
    y_labels: np.ndarray,
    target_distribution: str = 'balanced',
    augment: bool = True,
    augment_params: Dict = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oversample minority classes to balance the dataset.
    
    Args:
        X_sequences: Sequence data with shape (n_samples, window, features)
        y_labels: Labels with shape (n_samples,)
        target_distribution: 'balanced' or 'majority' 
                            - 'balanced': all classes have equal count
                            - 'majority': all classes match the majority class count
        augment: If True, augment duplicated samples instead of exact copies
        augment_params: Parameters for augmentation (if None, uses defaults)
        seed: Random seed for reproducibility
        
    Returns:
        X_oversampled, y_oversampled with balanced classes
    """
    np.random.seed(seed)
    random.seed(seed)
    
    if augment_params is None:
        augment_params = {
            'noise_level': 0.01,
            'scale_range': (0.95, 1.05),
            'shift_range': 5,
            'apply_noise': True,
            'apply_scaling': True,
            'apply_shift': True
        }
    
    # Get class distribution
    unique_classes, class_counts = np.unique(y_labels, return_counts=True)
    
    if target_distribution == 'balanced':
        # equal number of samples for all classes
        target_count = int(np.mean(class_counts))
    else:  # 'majority'
        # all classes have as many samples as the majority class
        target_count = int(np.max(class_counts))
    
    print(f"Oversampling to {target_distribution} distribution:")
    print(f"Target count per class: {target_count}")
    
    # Store oversampled data
    X_oversampled = []
    y_oversampled = []
    
    for cls in unique_classes:
        # Get samples for this class
        cls_mask = (y_labels == cls)
        cls_samples = X_sequences[cls_mask]
        cls_labels = y_labels[cls_mask]
        
        current_count = len(cls_samples)
        
        # Add all original samples
        X_oversampled.append(cls_samples)
        y_oversampled.append(cls_labels)
        
        # Calculate how many more samples are needed
        samples_needed = target_count - current_count
        
        if samples_needed > 0:
            print(f"  Class {cls}: {current_count} → {target_count} (+{samples_needed} samples)")
            
            # Randomly select samples to duplicate
            duplicate_indices = np.random.choice(
                len(cls_samples), 
                size=samples_needed, 
                replace=True
            )
            
            duplicates = cls_samples[duplicate_indices]
            
            if augment:
                # Apply augmentation to duplicates
                augmented_duplicates = []
                for sample in duplicates:
                    aug_sample = augment_time_series(sample, **augment_params)
                    augmented_duplicates.append(aug_sample)
                duplicates = np.array(augmented_duplicates)
            
            X_oversampled.append(duplicates)
            y_oversampled.append(np.full(samples_needed, cls))
        else:
            print(f"  Class {cls}: {current_count} (no oversampling needed)")
    
    # Concatenate all classes
    X_oversampled = np.concatenate(X_oversampled, axis=0)
    y_oversampled = np.concatenate(y_oversampled, axis=0)
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(X_oversampled))
    X_oversampled = X_oversampled[shuffle_idx]
    y_oversampled = y_oversampled[shuffle_idx]
    
    print(f"\nFinal dataset size: {len(X_oversampled)} samples")
    print("Final class distribution:")
    final_unique, final_counts = np.unique(y_oversampled, return_counts=True)
    for cls, count in zip(final_unique, final_counts):
        print(f"  Class {cls}: {count:5d} samples ({count/len(y_oversampled)*100:5.2f}%)")
    
    return X_oversampled, y_oversampled


def extract_time_features(
    df: pd.DataFrame,
    time_column: str = 'time',
    extract_hour: bool = True,
    extract_day_of_week: bool = True,
    extract_day_of_month: bool = True,
    use_cyclical_encoding: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract temporal features from a time column and optionally encode them cyclically.
    Cyclical encoding (sine/cosine) preserves the circular nature of time features
    
    Args:
        df: Input DataFrame with a time/timestamp column
        time_column: Name of the column containing time information
        extract_hour: Extract hour of day (0-23)
        extract_day_of_week: Extract day of week (0-6, Monday=0)
        extract_day_of_month: Extract day of month (1-31)
        use_cyclical_encoding: Use sine/cosine encoding for cyclical features
        verbose: Print information about extracted features
        
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    
    if verbose:
        print("=" * 80)
        print("TIME FEATURE ENGINEERING")
        print("=" * 80)
    
    # Check if time column exists
    if time_column not in df.columns:
        if verbose:
            print(f"⚠️  Time column '{time_column}' not found. Skipping time feature extraction.")
            print("=" * 80)
        return df
    
    num_features_added = 0
    
    # Position in sequence (normalized)
    if 'sample_index' in df.columns:
        df['time_position'] = df.groupby('sample_index')[time_column].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
        num_features_added += 1
        
        if use_cyclical_encoding:
            # Cyclical encoding of position
            df['time_position_sin'] = np.sin(2 * np.pi * df['time_position'])
            num_features_added += 1
            
            if verbose:
                print(f"✓ Added cyclical position features (sin/cos)")
    
    if extract_hour and use_cyclical_encoding:
        simulated_hour = (df[time_column] % 24.0)
        df['hour_sin'] = np.sin(2 * np.pi * simulated_hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * simulated_hour / 24.0)
        num_features_added += 2
        
        if verbose:
            print(f"✓ Added cyclical hour features (sin/cos)")
    
    if extract_day_of_week and use_cyclical_encoding:
        simulated_day = (df[time_column] % 7.0)
        df['day_of_week_sin'] = np.sin(2 * np.pi * simulated_day / 7.0)
        df['day_of_week_cos'] = np.cos(2 * np.pi * simulated_day / 7.0)
        num_features_added += 2
        
        if verbose:
            print(f"✓ Added cyclical day of week features (sin/cos)")
    
    if verbose:
        print(f"\nTotal time features added: {num_features_added}")
        print(f"New shape: {df.shape}")
        print("=" * 80)
    
    return df



