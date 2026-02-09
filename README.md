# Precog_CV_CNN
It's immensely easy to trick a neural network, let's make it smarter!

## Directory structure:
```c
Precog_CV_CNN/
├── images_archive
│   ├── adversaraial_attack.png
│   ├── color_map.png
│   ├── lazy_model_confusion.png
│   ├── lazy_model_feature_vis.png
│   ├── lazy_model_ideal_digit.png
│   ├── lazy_model_test_gradcam.png
│   ├── lazy_model_train_gradcam.png
│   ├── lazy_model_train_test.png
│   ├── lazy_model_wrong_samples.png
│   ├── normal_mnist.png
│   ├── robust_model_confusion.png
│   ├── robust_model_feature_vis.png
│   ├── robust_model_ideal_digit.png
│   ├── robust_model_test_gradcam.png
│   ├── robust_model_train_gradcam.png
│   ├── robust_model_wrong_samples.png
│   ├── test_5.png
│   ├── Test Set ('Hard')_biased_set.png
│   ├── training_5.png
│   └── Training Set ('Easy')_biased_set.png
├── precog_cnn.ipynb
├── failures_doc.pdf
├── report.pdf
├── unbiased_mnist.ipynb
├── presentation.pdf
└── README.md

2 directories, 22 files
```

## Commands to run project:
All the code is present in `precog_cnn.ipynb` with a dialogue to explain, introduce and motivate each cell in the notebook. Additionally, all plots generated are downloaded at the end into an archived zip file that can be extracted for further reference and inspection. The `unbiased_mnist.ipynb` notebook contains the same functions on the standard MNIST dataset for comparison.

Report and presentation were crafted using the afforementioned plots and LATEX.

## Dependencies:
All dependencies have been imported in the very beginning of `precog_cnn.ipynb` appropriately. For reference, 

```c
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
```

## Approach Details:
The approach has been stated in extreme detail in both the notebooks - `report.pdf` and `failures_doc.pdf`, with summary tables and visualizations as aids. 


## Approach: The Lazy Artist vs. Robust Learning

This project investigates how Convolutional Neural Networks (CNNs) tend to exploit "shortcuts" (like color) rather than learning fundamental features (like shape), and develops strategies to force structural learning.

### 1. Biased Canvas: Dataset Preparation
To simulate real-world data bias, we modified the MNIST dataset by injecting a controlled correlation between digit labels and specific colors.
* **Bias Injection:** 95% of training samples follow a fixed color-map (e.g., digit '0' is always Red).
* **Testing Strategy:** The test set uses "Hard" samples where colors are randomized or swapped, ensuring that a model relying on color shortcuts will fail.
* **Noise:** Added background textures and varying foreground strokes to test the model's ability to distinguish signal from noise.

### 2. Model Architectures
We utilized a custom 3-layer CNN architecture ("Lazy Model") to study the impact of bias:
* **Architecture:** 3 Convolutional layers (16 filters) followed by Adaptive Average Pooling.
* **Objective:** Compare a standard training loop against a **Robust Training** loop that employs a "Reduced Random Component" (95% probability randomization) to break the color-label correlation.

### 3. The Prober: Explainability & Visualization
To "interrogate" what the models actually learned, we employed several CV techniques:
* **Neuron Visualization:** Initializing random tensors and maximizing activations via `register_hook()` to see what features each layer detects.
* **Ideal Digit Synthesis:** Freezing weights and tuning image pixels to visualize the model's internal representation of a "perfect" digit.
* **Grad-CAM:** Generating heatmaps using gradient average pooling to prove that the Lazy Model ignores the digit and looks only at the background color.

### 4. The Intervention: Bias Mitigation
We explored six strategies to overcome the "Lazy Artist" effect:
1.  **Data Augmentation:** Using `ColorJitter`.
2.  **Forward Pass Edits:** Penalizing outlier weights.
3.  **Consistency Loss:** Forcing color-invariant predictions.
4.  **Color Randomization (Selected):** Introducing independent foreground/background randomization (Method 6), which achieved **75% test accuracy**.

### 5. Adversarial Robustness
We evaluated the Robust Model against **Adversarial Attacks** using the Fast Gradient Sign Method (FGSM).
* **Discovery:** The model breaks at a very low threshold ($\epsilon = 0.04$).
* **Key Insight:** Robustness to distribution bias (color) does not automatically translate to robustness against input perturbations (adversarial noise).

### Summary of Results
| Metric | Lazy Model | Robust Model |
| :--- | :--- | :--- |
| **Strategy** | Standard Training | Random Color Warp |
| **Accuracy (Biased Test)** | Low/Failing | **75%** |
| **Feature Focus** | Background/Color | Structural Curves |
| **Explainability** | Meaningless Blobs | Identifiable Digits |
