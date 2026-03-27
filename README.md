# Advanced Neural Networks and Deep Learning Challenges - Politecnico di Milano

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TorchVision](https://img.shields.io/badge/TorchVision-0.15+-teal.svg)](https://pytorch.org/vision/)

**Team**: ReLUctant  
**Authors**: Alessandro Del Fatti, Matteo Garzone, Francesco Genovese  
**Course**: Applied Neural Networks and Deep Learning (AN2DL), Politecnico di Milano, 2025 
**Grade**: 10/10 

> [!NOTE]
> Solutions for two machine learning competitions showcasing advanced deep learning techniques in time series classification and biomedical image segmentation.

---

## 🎯 Challenge 1: Pirates Pain Classification (Time Series Classification)

### Objective

Multiclass time series classification to predict pain levels (No Pain, Low Pain, High Pain) based on biomechanical sensor data from pirates.

### Dataset Characteristics

- Severe class imbalance.
- High-dimensional temporal data requiring temporal dependency capture

### Key Achievements

- **Bidirectional LSTM**: 0.9472 validation F1
- **Bidirectional GRU**: 0.8651 validation F1
- **Bidirectional RNN**: 0.8539 validation F1
- Hierarchical classification approach explored as promising direction
- Successfully addressed class imbalance through targeted techniques

### Data Profiling & Feature Engineering

- Applied **correlation-based dimensionality reduction**
- **Autocorrelation analysis** revealed strong temporal dependencies

### Preprocessing Pipeline

- **Normalization**: Min-Max normalization
- **Temporal Encoding**: Cyclical time-position encoding using sine/cosine features
- **Sequence Windowing**: Sliding window strategy with overlapping segments

### Techniques Applied

**Data Augmentation:**

- Inverse-proportional class weights for loss function weighting
- Oversampling of minority classes

**Loss Functions:**

- Weighted categorical cross-entropy
- Label Smoothing loss (custom implementation)
- Focal loss for class imbalance handling

**Regularization:**

- Dropout at multiple layers
- L2 regularization
- Label smoothing
- Gradient clipping and normalization

**Optimization:**

- Optimizer: AdamW
- Dynamic learning rate scheduling
- Early stopping on validation F1
- Adaptive learning rate reduction

**Architecture Enhancements:**

- Added 1D Convolutional layer
- Added Attention layer for improved performance
- Hierarchical classification approach (binary classifiers in cascade)

### Files & Notebooks

- [Main Pipeline Notebook](challenge-1/pirates_pain_classification.ipynb) - Complete training and evaluation workflow
- [Data Profiling](challenge-1/data_profiling.ipynb) - Exploratory data analysis
- [Configurations](challenge-1/configs/) - Model hyperparameter configs (YAML format)
- [Trained Models](challenge-1/models/) - Best performing model checkpoints
- [Source Code](challenge-1/src/) - Modular pipeline components

---

## 📂 Project Structure

```
polimi-an2dl/
├── challenge-1/               # Time Series Classification
│   ├── src/                   # Modular pipeline code
│   ├── configs/               # Experiment configurations
│   ├── data/                  # Raw and processed datasets
│   ├── models/                # Trained checkpoints
│   ├── logs/                  # Grid search logs
│   ├── pirates_pain_classification.ipynb
│   ├── data_profiling.ipynb
│   └── requirements.txt
├── challenge-2/               # Image Classification
│   ├── utils/
│   ├── configs/
│   ├── train_data/
│   ├── test_data/
│   ├── models/
│   ├── challenge-2.ipynb
│   ├── experiments.ipynb
│   └── README.md
└── .venv/                     # Python virtual environment
```

---

## 🖼️ Challenge 2: Grumpy Doctogres (Image Classification)

### Objective

Classify low-magnification Whole Slide Images (WSIs) into four molecular subtypes associated with breast cancer.

### Dataset Characteristics

- Each sample contains: image and mask delimiting cancerous tissue area
- Class imbalance: Luminal A and Luminal B overrepresented, Triple Negative underrepresented
- Presence of contaminated slides with artifacts (marker ink, greenish anomalies)

### Key Achievements

- **CNN Baselines**: 0.2343 public test F1
- **Foundation Model CNNs**: 0.2706 public test F1
- **Patch-Based Training**: 0.3310 validation F1
- **MIL Transformer**: 0.3650 public test F1
- **K-Fold MIL Transformer (Final)**: 0.3963 public test F1 with 0.4210 ± 0.067 validation F1

### Data Cleaning & Analysis

- Custom filtering algorithm to detect and reject contaminated slides automatically
- Identified and removed images compromised by significant artifacts

### Preprocessing Pipeline

**Normalization approaches explored:**

- Macenko Normalization combined with Araujo Contrast Stretching (mitigates staining variations)
- Z-score normalization during embedding phase (raw data strategy)
- Comparison performed due to artifacts amplification concerns in sparse regions

**Patch Extraction:**

- Mask-guided extraction at multiple resolutions
- Strict slide-level data partitioning to prevent data leakage
- Tumor-content filtering based on mask annotation
- Overlapping patches for training augmentation
- Non-overlapping patches for validation

### Techniques Applied

**Data Augmentation:**

- Stochastic CutMix patch extraction
- Feature Bags MixUp (same-class and same-slide patches)
- SSL-based invariance via Phikon foundation model (renders additional augmentation redundant)

**Loss Functions:**

- Weighted categorical cross-entropy
- Label Smoothing loss
- Focal loss for class imbalance
- Contrastive Loss for self-supervised auxiliary learning

**Optimization & Regularization:**

- Optimizers compared: AdamW vs Lion
- Dropout regularization
- L1 and L2 regularization
- Learning rate warmup strategy
- Adaptive learning rate schedulers

**Final Architecture:**

- **Feature Extractor**: Phikon foundation model
- **Auxiliary Features**: Geometric mask encoder
- **Aggregation**: Transformer encoder with multi-head attention
- **Pooling**: Gated attention mechanism
- **Stabilization**: Layer normalization throughout

**Validation Strategy:**

- Stratified k-fold cross-validation at slide level
- Ensemble aggregation of fold models with soft-averaging predictions

### Files & Notebooks

- [Main Pipeline Notebook](challenge-2/challenge-2.ipynb) - Primary training and evaluation
- [Experiments](challenge-2/experiments.ipynb) - Exploratory experiments and ablation studies
- [Utilities](challenge-2/utils/) - Helper functions and preprocessing
- [Configurations](challenge-2/configs/) - Model hyperparameter settings
- [Trained Models](challenge-2/models/) - Model checkpoints and weights

---

## ⚗️ Methodological Approach

Both challenges follow structured experimental pipelines:

**Challenge 1 (Time Series):**

- Baseline model exploration (RNN, LSTM, GRU variants)
- Hyperparameter grid search across multiple phases
- Architecture enhancements (1D CNN, Attention layers)
- Hierarchical classification exploration

**Challenge 2 (Image Classification):**

- Progressive architecture evolution from CNNs to foundation models to MIL
- Patch-level feature extraction and aggregation
- K-fold cross-validation with ensemble predictions
- Comparative analysis of normalization and optimization strategies

---

## 📊 Experimental Results

### Challenge 1 - Time Series Classification

| Model       | Validation F1 |
| ----------- | ------------- |
| RNN         | 0.6294        |
| Bi-RNN      | 0.8539        |
| LSTM        | 0.6294        |
| **Bi-LSTM** | **0.9472**    |
| GRU         | 0.6294        |
| Bi-GRU      | 0.8651        |

**Baseline Selection**: Bidirectional LSTM selected as primary architecture due to consistent superior performance across hyperparameter configurations.

### Challenge 2 - Image Classification (WSI)

| Experiment Category        | Validation F1      | Public Test F1 |
| -------------------------- | ------------------ | -------------- |
| CNN-Based Experiments      | 0.2315 ± 0.018     | 0.2343         |
| Pretrained Foundation CNNs | 0.2640 ± 0.027     | 0.2706         |
| Patch-Based Training       | 0.3310 ± 0.023     | 0.3038         |
| MIL Transformer Learning   | 0.3695 ± 0.016     | 0.3650         |
| **K-Fold MIL Transformer** | **0.4210 ± 0.067** | **0.3963**     |

---

## 📝 Reports & Documentation

- **Challenge 1 Report**: [challenge-1/report.pdf](challenge-1/report.pdf) - Detailed analysis of time series classification approach
- **Challenge 2 Report**: [challenge-2/report.pdf](challenge-2/report.pdf) - Comprehensive study of MIL-based WSI classification

---

## 💾 Technologies

- **Deep Learning**: PyTorch 2.0+, TorchVision
- **Foundation Models**: Phikon
- **Data**: NumPy, Pandas, Scikit-learn, SciPy
- **Visualization**: Matplotlib, Seaborn, TensorBoard
- **Experiment Tracking**: TensorBoard.

---
