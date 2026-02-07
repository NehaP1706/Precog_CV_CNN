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
├── presentation.pdf
└── README.md

2 directories, 22 files
```

## Commands to run project:
All the code is present in `precog_cnn.ipynb` with a dialogue to explain, introduce and motivate each cell in the notebook. Additionally, all plots generated are downloaded at the end into an archived zip file that can be extracted for further reference and inspection.

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

