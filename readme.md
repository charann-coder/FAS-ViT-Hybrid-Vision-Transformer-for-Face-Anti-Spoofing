# FAS-ViT: Hybrid Vision Transformer for Face Anti-Spoofing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

A production-ready **Face Anti-Spoofing (FAS)** framework featuring a novel hybrid architecture that fuses a **Lightweight Vision Transformer** with multi-scale convolutional features. This project achieves **82.18% calibrated accuracy** and an **85.93% AUC score**, demonstrating robust performance against presentation attacks.

---

### ✨ Live Demo
*A real-time demonstration of the `live_inference.py` script detecting a spoof attack.*

![Live Demo GIF](https://your-gif-url-here.com/demo.gif)
*(**Recommendation**: Record a short GIF of the live demo and replace the URL above to significantly increase the impact of your repository.)*

---

## 📖 Table of Contents
- [Performance Highlights](#-performance-highlights)
- [Architecture Overview](#️-architecture-overview)
- [Advanced Features](#-advanced-features)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Repository Structure](#-repository-structure)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## 📊 Performance Highlights

The best model (`epoch_19.pth`) was selected based on a balance of accuracy and error rates, demonstrating superior generalization over models with higher validation accuracy alone.

**Results on the LCC-FASD Test Set:**

| Metric                      | Raw (thr=0.5) | Calibrated (thr=0.4767) |
| --------------------------- | :-----------: | :---------------------: |
| **Accuracy**                |    81.20%     |        **82.18%**       |
| **Balanced Accuracy**       |    79.22%     |        **79.58%**       |
| **AUC Score**               |   **85.93%**  |           --            |
| **ACER** (Avg. Error Rate)  |    20.78%     |        **20.42%**       |
| **EER** (Equal Error Rate)  |   **21.97%**  |           --            |

#### Visualizations
| Raw Confusion Matrix                               | Calibrated Confusion Matrix                        | ROC Curve                                  |
| -------------------------------------------------- | -------------------------------------------------- | ------------------------------------------ |
| ![Raw CM](results/plots/test_confusion_matrix_raw.png) | ![Calibrated CM](results/plots/test_confusion_matrix_calibrated.png) | ![ROC Curve](results/plots/test_roc_curve.png) |

---

## 🏗️ Architecture Overview

The **FASViTClassifier** employs a sophisticated hybrid approach to capture both local-texture and global-spatial artifacts indicative of spoofing attacks.

-   **Multi-Scale Feature Extractor**: A convolutional backbone fuses features from three parallel branches:
    1.  **Xception Blocks**: Efficiently learn rich features using depthwise separable convolutions.
    2.  **ASPP (Atrous Spatial Pyramid Pooling)**: Captures contextual information at multiple scales to handle variations in object size.
    3.  **Laplacian Frequency Branch**: Explicitly analyzes high-frequency details to detect subtle artifacts common in print and replay attacks.
-   **Lightweight Vision Transformer (ViT)**: The fused feature map is processed by a shallow ViT encoder (1-2 layers). Its self-attention mechanism models global spatial relationships between different facial regions, which is crucial for identifying inconsistencies in complex spoofing attacks.
-   **Multi-Task Classification Head**: The model is trained on two simultaneous tasks for improved feature learning:
    1.  **Primary Task**: Binary classification (Real vs. Spoof).
    2.  **Auxiliary Task**: Multi-class classification of spoof types (e.g., Print, Replay, Mask).

---

## 🌟 Advanced Features

-   **Self-Supervised Pre-training**: The model can be pre-trained on a self-supervised task (rotation prediction) to learn robust, generalized feature representations before being fine-tuned on the main anti-spoofing task.
-   **Progressive Unfreezing**: During training, layers are unfrozen gradually, allowing the model to stabilize and learn more effectively.
-   **Adaptive Thresholding**: The live demo includes an experimental adaptive thresholding mechanism that adjusts to lighting conditions and uses temporal smoothing for more stable predictions.
-   **Configurable Training Profiles**: Easily switch between `speed` and `accuracy` profiles, which adjust patch size, transformer depth, and learning rates for different deployment needs.

---

## 🚀 Quick Start

Get the project up and running in a few simple steps.

#### 1. Clone the Repository
```bash
git clone https://github.com/charann-coder/fas.git
cd fas
```

#### 2. Create Environment & Install Dependencies
We recommend using **Miniconda** for environment management.
```bash
# Create and activate a new conda environment
conda create -n fas-vit python=3.8 -y
conda activate fas-vit

# Install all required packages
pip install -r requirements.txt
pip install kaggle
```

#### 3. Download and Prepare the Dataset
This project uses the **LCC-FASD dataset**, which can be downloaded directly using the Kaggle API.

1.  **Set up your Kaggle API credentials.**
    -   Log in to your Kaggle account and go to your account settings page (`https://www.kaggle.com/account`).
    -   Click `Create New API Token` to download a `kaggle.json` file.
    -   Place this file in the required location: `C:\Users\<Your-Username>\.kaggle\kaggle.json` on Windows, or `~/.kaggle/kaggle.json` on Linux/macOS.

2.  **Download and unzip the dataset** into the project directory by running this command:
    ```bash
    kaggle datasets download -d faber24/lcc-fasd -p . --unzip
    ```
    This will create a folder named `LCC_FASD` in your project root.

3.  **(CRITICAL) Rename and Restructure the Dataset.**
    The training scripts expect the data to be in `train`, `val`, and `test` folders. You must rename the unzipped directories to match this structure.

    -   Rename `LCC_FASD` to `LCC_dataset`.
    -   Inside `LCC_dataset`, rename the subfolders as follows:
        -   `LCC_FASD_training`   -> `train`
        -   `LCC_FASD_development` -> `val`
        -   `LCC_FASD_evaluation`   -> `test`

    Your final directory structure **must** look like this for the code to work:
    ```
    fas-vit/
    ├── LCC_dataset/
    │   ├── train/
    │   │   ├── real/
    │   │   └── spoof/
    │   ├── val/
    │   │   ├── real/
    │   │   └── spoof/
    │   └── test/
    │       ├── real/
    │       └── spoof/
    ├── checkpoints/
    ├── configs/
    └── ... (rest of the project files)
    ```

#### 4. Run the Live Demo
Run the real-time face anti-spoofing demo using your webcam and the provided pre-trained model.
```bash
python live_inference.py --ckpt checkpoints/epoch_19.pth
```
**Interactive Controls:**
-   `q`: Quit the application.
-   `s`: Save the current frame to the `captures/` directory.
-   `t` / `g`: Increase/decrease the decision threshold.

---

## 📖 Usage Guide

#### Training
The model can be trained directly or by using a two-stage SSL pre-training approach.

**Option 1: Direct Fine-Tuning**
```bash
# Use the default 'speed' profile
python train.py

# Or specify the 'accuracy' profile
TRAIN_PROFILE=accuracy python train.py
```

**Option 2: Self-Supervised Pre-training + Fine-tuning**
```bash
# Step 1: Run SSL pre-training (requires ssl_data.csv)
python train_ssl.py

# Step 2: Run fine-tuning (automatically loads the best SSL checkpoint)
python train.py
```

#### Evaluation
Evaluate the model's performance on the test set and generate metrics.
```bash
# Evaluate using the best checkpoint
python test.py --ckpt checkpoints/epoch_19.pth

# Optimize for the minimum ACER score during evaluation
python test.py --ckpt checkpoints/epoch_19.pth --mode min_acer
```

---

## 📁 Repository Structure

```
.
├── checkpoints/
│   └── epoch_19.pth        # Best performing model checkpoint
├── configs/
│   └── train_config.yaml   # Training configuration
├── results/
│   ├── plots/              # Saved plots (CM, ROC)
│   ├── test_metrics_log.csv
│   └── test_predictions.csv
├── .gitignore
├── live_inference.py       # Script for live webcam demo
├── model.py                # FASViTClassifier architecture definition
├── README.md
├── requirements.txt
├── test.py                 # Script for model evaluation
└── train.py                # Script for model training
```

---

## 🤝 Contributing
Contributions are highly welcome! Please fork the repository and submit a pull request with your proposed changes.

---

## 📚 Citation
If you use this code or find our work helpful in your research, please consider citing it. As this is unpublished research, please cite the repository directly.

```bibtex
@misc{fas-vit-2025,
  author       = {Y Sri Charan},
  title        = {FAS-ViT: Hybrid Vision Transformer for Face Anti-Spoofing},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/charann-coder/fas}}
}
```

---

## 📄 License
This project is currently licensed as **All Rights Reserved – Research in Progress**.

Copyright (c) 2025 Y Sri Charan

This code and associated files are part of ongoing, unpublished research and are made available for academic review only. Redistribution, modification, or use of this code in any form is not permitted without explicit written permission from the author.

Once the related research is formally published, this repository may be updated with a permissive open-source license (e.g., MIT).