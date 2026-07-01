# AOSLO_Cone_Counting
Cone counting in AOSLO images


**Lightweight Deep Learning Models for Automated Retinal Cell Counting**

[![License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Data License](https://img.shields.io/badge/Data-CC%20BY--NC%204.0-lightgrey.svg)](data/README.md)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)

---

## Abstract

This repository accompanies the manuscript **"Unified End-to-End Density-Based Counting for AOSLO Cone Photoreceptor Quantification"** submitted to MICCAI 2026.

We propose and evaluate **four lightweight deep learning architectures** for automated cone density estimation in Adaptive Optics Scanning Laser Ophthalmoscopy (AOSLO) images. Our central hypothesis is that carefully designed lightweight models can improve accuracy over complex baselines while dramatically reducing computational requirements.

**Key Results:**
- **74% parameter reduction** (538K vs 2.06M parameters)
- **40% lower MAE** (921.6 vs 1534.6 cones)
- **60% smaller model size** (6.4 MB vs 16 MB)

This enables earlier and more accurate diagnosis of genetic eye diseases such as **retinitis pigmentosa** and **Stargardt's disease**.

---

## Results

### Performance Comparison

| Model | MAE (cones) | RMSE | MAPE (%) | Parameters | Size |
|-------|-------------|------|----------|------------|------|
| Baseline (CoDE) | 1534.57 | - | - | 2.06M | 16.0 MB |
| **Model A** | **921.62** | 1255.19 | 13.64 | 538K | 6.4 MB |
| Model B | 977.05 | 1253.57 | 12.80 | 538K | 6.4 MB |
| Model C | 3884.54 | 4970.76 | 55.29 | ~540K | 6.4 MB |
| Model D | 1043.95 | 1317.45 | 17.08 | 380K | 3.6 MB |

> **Model A achieves 40% better accuracy with 74% fewer parameters**

### Model Recommendations

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| Best accuracy | Model A | Lowest MAE (921.62 cones) |
| Spatial interpretation | Models A/B | Density map output for clinical validation |
| Resource-constrained | Model D | Smallest size (3.6 MB, 380K params) |
| Systematic bias correction | Model B | Linear correction reduces bias |

---

## Project Structure

```
AOSLO-Cell-Density-Estimation/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── LICENSE                   # Apache 2.0 license
│
├── notebooks/                # Training & evaluation notebooks
│   ├── README.md
│   ├── Baseline_Model.ipynb
│   ├── Model_A_Lightweight.ipynb
│   ├── Model_B_LinearRegression_with_Metrics.ipynb
│   ├── Model_C_GlobalSum.ipynb
│   └── Model_D_DirectRegression.ipynb
│
├── models/                   # Pre-trained model weights
│   ├── README.md
│   ├── Model_A.h5           # Lightweight U-Net (6.4 MB)
│   ├── ModelC.h5            # Global-Sum U-Net (6.4 MB)
│   ├── ModelD.h5            # Direct Regression (3.6 MB)
│   └── Baseline_Model.keras # CoDE reference (16 MB)
│
├── data/                     # Dubis dataset (264 AOSLO images)
│   ├── README.md
│   ├── Training+Density/    # 184 images + density maps
│   └── Validation+Density/  # 80 images + density maps
│
├── paper/                    # Thesis LaTeX source & PDF
│   ├── main.tex
│   ├── bibliography.bib
│   └── attachments/
│
└── resources/                # Additional documentation
    └── AOSLO_DeepLearning_Tesis_Final.pdf
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/nicolaycz/AOSLO-Cell-Density-Estimation.git
cd AOSLO-Cell-Density-Estimation
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```python
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

---

## Quick Start

### Using Pre-trained Models

```python
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the best model (Model A)
model = load_model('models/Model_A.h5')

# Load and preprocess an AOSLO image
img = Image.open('data/Validation+Density/image001.tif')
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)  # Shape: (1, 256, 256, 3)

# Predict density map
density_map = model.predict(img)

# Calculate total cone count
cone_count = np.sum(density_map) / 100
print(f"Estimated cone count: {cone_count:.0f}")
```

### Training from Scratch

1. Launch Jupyter: `jupyter notebook`
2. Open `notebooks/Model_A_Lightweight.ipynb`
3. Update data paths for local execution:

```python
# Change from Colab paths:
train_path = '/content/drive/MyDrive/.../Training+Density/'
# To local paths:
train_path = '../data/Training+Density/'
```

4. Run all cells

---

## Model Architectures

### Model A - Lightweight U-Net (Recommended)

```
Input (256x256x3)
    ↓
Encoder: Conv(16) → Conv(32) → Conv(64)
    ↓
Bottleneck: Conv(128) + Dropout(0.5)
    ↓
Decoder: Conv(64) → Conv(32) → Conv(16)
    ↓
Output: Density Map (256x256x1)
```

- **Loss**: Pixel-wise MSE on density maps
- **Output**: Spatial density map (interpretable)
- **Count**: `sum(density_map) / 100`

### Model D - Direct Regression (Most Efficient)

```
Input (256x256x3)
    ↓
Encoder: Conv(16) → Conv(32) → Conv(64)
    ↓
Bottleneck: Conv(128) + Dropout(0.5)
    ↓
GlobalAveragePooling2D
    ↓
Dense(64) → Dense(1)
    ↓
Output: Scalar Cone Count
```

- **Loss**: MSE on counts
- **Output**: Direct count (no spatial info)
- **Efficiency**: 77% smaller than U-Net models

---

## Dataset

The **Dubis dataset** contains 264 split-detector AOSLO images:

| Condition | Samples |
|-----------|---------|
| Control (healthy) | 60 |
| Stargardt's disease | 65 |
| Retinitis pigmentosa | 139 |

See [`data/README.md`](data/README.md) for detailed documentation.


## License

- **Code**: [Apache License 2.0](LICENSE)
- **Data**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Non-commercial research only)

---

## Disclaimer

This software is provided "as is" for **research purposes only**. It has not been approved for clinical diagnostics and must not be used to guide patient care without appropriate validation and regulatory clearance.
