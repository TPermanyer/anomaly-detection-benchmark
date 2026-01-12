<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/PyOD-Anomaly_Detection-green?style=for-the-badge" alt="PyOD">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<h1 align="center">Anomaly Detection Benchmarking Platform</h1>

<p align="center">
  <b>An interactive, modular web application for comparing and evaluating multiple unsupervised anomaly detection algorithms with comprehensive metrics, visualizations, and explainability features.</b>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Implemented Methods](#implemented-methods)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Metrics & Evaluation](#metrics--evaluation)
- [Extensibility](#extensibility)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

This project is a **Final Degree Project (TFG)** that provides a comprehensive platform for benchmarking unsupervised anomaly detection algorithms. The application allows users to:

- Upload their own datasets (CSV, Parquet, MAT formats)
- Select and configure multiple anomaly detection methods
- Evaluate models using cross-validation with multiple metrics
- Visualize results through interactive charts and projections
- Explain model predictions using SHAP values

The platform is designed with **modularity** in mind, making it easy to add new detection methods, preprocessing pipelines, or evaluation metrics.

---

## Features

### Multi-Algorithm Comparison
Compare 11+ anomaly detection algorithms side-by-side with unified evaluation metrics.

### Rich Visualizations
- **Precision-Recall Curves** for ranking quality assessment
- **Confusion Matrices** with adjustable thresholds
- **Score Distributions** with class separation analysis
- **2D Projections** (PCA & t-SNE) for data exploration
- **Method Correlation Heatmaps** to understand algorithm similarities

### Flexible Preprocessing
- **Dense Mode**: AutoPreprocessor with imputation, scaling, and log transforms
- **Sparse Mode**: One-Hot Encoding with SVD dimensionality reduction
- **Minimal Mode**: For tree-based methods that handle categorical data natively

### Explainability
Integrated **SHAP (SHapley Additive exPlanations)** for understanding feature importance and model decisions.

### Ensemble Methods
Combine multiple detectors using **Mean Percentile Ensemble** for more robust predictions.

### Performance Optimization
- Data subsampling for large datasets
- Efficient cross-validation with progress tracking
- Execution time benchmarking

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT WEB APP                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │  Data Tab    │ │ Analysis Tab │ │     Deep Dives Tab       │ │
│  │  - Upload    │ │ - Run CV     │ │  - PR Curve              │ │
│  │  - Schema    │ │ - Leaderboard│ │  - Confusion Matrix      │ │
│  │  - Preproc   │ │ - Correlation│ │  - SHAP Explainability   │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CORE LIBRARY                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Methods   │  │   Metrics   │  │       Evaluate          │  │
│  │  - IForest  │  │  - AP       │  │  - Cross-Validation     │  │
│  │  - LOF      │  │  - AUC-ROC  │  │  - Scoring Pipeline     │  │
│  │  - OCSVM    │  │  - F1       │  │                         │  │
│  │  - DeepSVDD │  │  - MCC      │  │                         │  │
│  │  - ...      │  │  - P@K      │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implemented Methods

| Method | Type | Description |
|--------|------|-------------|
| **Isolation Forest** | Tree-based | Isolates anomalies using random feature splits |
| **LOF** | Density-based | Local Outlier Factor using k-NN density estimation |
| **One-Class SVM** | Boundary-based | Learns a decision boundary around normal data |
| **PCA Reconstruction** | Reconstruction | Detects anomalies via reconstruction error |
| **HBOS** | Histogram-based | Fast histogram-based outlier detection |
| **ECOD** | Statistical | Empirical Cumulative Distribution-based detection |
| **COPOD** | Statistical | Copula-based outlier detection |
| **KNN** | Distance-based | K-Nearest Neighbors distance-based detection |
| **SDO** | Observer-based | Sparse Data Observers using K-Means centroids |
| **AutoEncoder** | Deep Learning | Neural network reconstruction error |
| **DeepSVDD** | Deep Learning | Deep Support Vector Data Description |
| **Ensemble** | Meta-method | Mean Percentile combination of multiple methods |

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip or conda package manager

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/anomaly-detection-benchmark.git
cd anomaly-detection-benchmark
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app/streamlit_app.py
```

5. **Open in browser**
Navigate to `http://localhost:8501`

---

## Usage

### 1. Upload Your Data
- Supported formats: **CSV**, **Parquet**, **MAT** (MATLAB)
- Select your **label column** (optional, for supervised evaluation)
- Choose **feature columns** for anomaly detection

### 2. Configure Pipeline
- **Feature Engineering**: Optional polynomial features or PCA reduction
- **Preprocessing Mode**: Dense, Sparse, or Minimal
- **Algorithm Selection**: Pick one or more detection methods
- **Ensemble Option**: Combine multiple selected methods

### 3. Run Evaluation
- Configure **K-Fold Cross-Validation** splits
- Click **"Run Evaluation"** to train and evaluate all models
- View the **Multi-metric Leaderboard** with color-coded best performers

### 4. Deep Dive Analysis
- Inspect individual methods with PR curves
- Adjust threshold for confusion matrix analysis
- Generate 2D projections (PCA/t-SNE)
- Compute SHAP values for explainability

---

## Project Structure

```
tfg-anomalies/
├── app/
│   ├── streamlit_app.py      # Main Streamlit application
│   └── utils/
│       ├── data.py           # Data loading and parsing utilities
│       ├── metrics.py        # Metric computation helpers
│       └── ui.py             # UI components and styling
│
├── core/
│   ├── methods/
│   │   ├── registry.py       # Method registry and factory functions
│   │   ├── base.py           # Base class for anomaly detectors
│   │   ├── isolation_forest.py
│   │   ├── lof.py
│   │   ├── ocsvm.py
│   │   ├── pca_recon.py
│   │   ├── hbos.py
│   │   ├── ecod.py
│   │   ├── copod.py
│   │   ├── knn.py
│   │   ├── sdo.py            # Sparse Data Observers
│   │   ├── deep_svdd.py
│   │   ├── autoencoder.py
│   │   ├── ensemble.py       # Mean Percentile Ensemble
│   │   ├── preproc_wrapper.py
│   │   └── sparse_auto.py    # Sparse preprocessing pipeline
│   │
│   ├── metrics/
│   │   └── ranking.py        # Ranking metrics (AP, P@K, PR curves)
│   │
│   ├── evaluate/
│   │   └── cv.py             # Cross-validation utilities
│   │
│   └── datasets/             # Dataset utilities (optional)
│
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## Metrics & Evaluation

| Metric | Range | Description |
|--------|-------|-------------|
| **AP (Average Precision)** | 0-1 | Area under the Precision-Recall curve |
| **AUC-ROC** | 0-1 | Area under the ROC curve |
| **AUC-PR** | 0-1 | Area under the Precision-Recall curve |
| **F1 Score** | 0-1 | Harmonic mean of Precision and Recall |
| **Precision** | 0-1 | True positives / (True + False positives) |
| **Recall** | 0-1 | True positives / (True positives + False negatives) |
| **MCC** | -1 to 1 | Matthews Correlation Coefficient |
| **Balanced Accuracy** | 0-1 | Average of recall per class |

---

## Extensibility

### Adding a New Detection Method

1. Create a new file in `core/methods/` (e.g., `my_method.py`):

```python
from .base import AnomalyDetector, ensure_2d
import numpy as np

class MyMethod(AnomalyDetector):
    def __init__(self, param1=10, contamination=0.1):
        self.param1 = param1
        self.contamination = contamination
        
    def fit(self, X, y=None):
        X = ensure_2d(np.asarray(X))
        # Your training logic here
        return self
    
    def score(self, X):
        X = ensure_2d(np.asarray(X))
        # Return anomaly scores (higher = more anomalous)
        return scores
```

2. Register in `core/methods/registry.py`:

```python
from .my_method import MyMethod

REGISTRY["MyMethod"] = lambda **kw: MyMethod(**kw)

DEFAULT_KWARGS["MyMethod"] = {"param1": 10, "contamination": 0.1}
```

3. The method will automatically appear in the UI!

---

## Screenshots

> *Add screenshots of your application here*

| Data Upload | Leaderboard | Deep Dive |
|-------------|-------------|-----------|
| ![Data](screenshots/data.png) | ![Leaderboard](screenshots/leaderboard.png) | ![DeepDive](screenshots/deepdive.png) |

---

## Future Improvements

- [ ] **Hyperparameter Optimization**: Integration with Optuna for automated tuning
- [ ] **Streaming Data Support**: Real-time anomaly detection for time series
- [ ] **Model Export**: Save trained models for production deployment
- [ ] **Additional Methods**: SUOD, LUNAR, LUNAR, etc.
- [ ] **Docker Support**: Containerized deployment
- [ ] **API Endpoints**: REST API for programmatic access

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Author

**Your Name**
- GitHub: [@your_username](https://github.com/your_username)
- LinkedIn: [Your Name](https://linkedin.com/in/your_profile)

---

<p align="center">
  <b>If you find this project useful, please consider giving it a star!</b>
</p>
