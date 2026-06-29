# MNIST Image Reconstruction using Autoencoders

> A production-quality, modular PyTorch implementation comparing three autoencoder architectures for unsupervised MNIST image reconstruction.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Comparison](#architecture-comparison)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Dataset Setup](#dataset-setup)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Configuration](#configuration)
10. [Testing](#testing)
11. [Design Decisions](#design-decisions)
12. [Future Improvements](#future-improvements)
13. [References](#references)
14. [License](#license)

---

## Project Overview

This project implements and compares three autoencoder architectures for reconstructing MNIST handwritten digit images from the [Kaggle MNIST dataset](https://www.kaggle.com/datasets/awsaf49/mnist-dataset):

| Architecture | Decoder Strategy | Key Characteristic |
|---|---|---|
| **FFNN** | Fully-connected layers | Baseline; ignores spatial structure |
| **CNN + Transposed Conv** | `ConvTranspose2d` | Learned upsampling |
| **CNN + Upsampling** | `Upsample` + `Conv2d` | Smooth upsampling without checkerboard artefacts |

All three architectures share a **latent dimension of 32**, giving a **24.5× compression ratio** relative to the 784-pixel input.

Key engineering features:
- Every hyperparameter lives in `configs/config.yaml` — zero hardcoded values in source files.
- A reusable `Trainer` class supports all three architectures without modification.
- Automatic GPU / MPS / CPU detection.
- Early stopping, gradient clipping, and learning-rate scheduling out of the box.
- Full evaluation suite: MSE, PSNR, SSIM, inference time, model size, and latent-space visualisations.

---

## Architecture Comparison

### Feed-Forward Autoencoder (FFNN)

```
Input (784)
  └─ Encoder: 784 → 512 → 256 → 128 → 64 → 32  (BatchNorm + ReLU)
       └─ Latent (32)
           └─ Decoder: 32 → 64 → 128 → 256 → 512 → 784  (BatchNorm + ReLU → Sigmoid)
                └─ Output (1×28×28)
```

- Weight init: Normal(0, 0.01); biases = 0
- Treats images as flat vectors; fast but ignores spatial relationships

### CNN Autoencoder — Transposed Convolutions

```
Input (1×28×28)
  └─ Encoder: Conv2d ×3 (stride=2) → 14×14 → 7×7 → 4×4 → Linear → Latent (32)
       └─ Decoder: Linear → 4×4 → ConvTranspose2d ×3 (stride=2) → 28×28
            └─ Output (1×28×28, Sigmoid)
```

- Learns spatial feature hierarchies
- May produce minor checkerboard artefacts at stride boundaries

### CNN Autoencoder — Upsampling + Conv

```
Input (1×28×28)
  └─ Encoder: (identical to CNN-Transpose)
       └─ Decoder: Linear → 4×4 → [Upsample(×2) + Conv2d] ×3 → 28×28
            └─ Output (1×28×28, Sigmoid)
```

- Replaces `ConvTranspose2d` with bilinear `Upsample` + `Conv2d`
- Smoother reconstructions without checkerboard artefacts

---

## Project Structure

```
mnist-autoencoder/
│
├── configs/
│   └── config.yaml              # All hyperparameters & paths
│
├── data/                        # CSV files go here (not tracked by git)
│
├── datasets/
│   ├── __init__.py
│   └── mnist_dataset.py         # MNISTCSVDataset + DataLoader factory
│
├── models/
│   ├── __init__.py
│   ├── ffnn_autoencoder.py
│   ├── cnn_transpose_autoencoder.py
│   └── cnn_upsampling_autoencoder.py
│
├── training/
│   ├── __init__.py
│   └── trainer.py               # Reusable Trainer + EarlyStopping
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # MSE, PSNR, SSIM, timing, compression
│   └── evaluator.py             # Full evaluation + plot pipeline
│
├── visualization/
│   ├── __init__.py
│   └── plots.py                 # EDA + training + reconstruction plots
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                # Colored console + rotating file logging
│   └── helpers.py               # Config, seed, device, checkpoint utils
│
├── outputs/
│   ├── checkpoints/             # Best model + epoch checkpoints
│   ├── reconstructions/         # Saved reconstruction images
│   ├── plots/                   # All generated PNG figures
│   └── logs/                    # Training history CSV + log files
│
├── tests/
│   └── test_project.py          # 37-test pytest suite
│
├── train.py                     # Training entry-point
├── evaluate.py                  # Evaluation entry-point
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/mnist-autoencoder.git
cd mnist-autoencoder

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

This project uses the Kaggle dataset [`awsaf49/mnist-dataset`](https://www.kaggle.com/datasets/awsaf49/mnist-dataset), **not** `torchvision.datasets.MNIST`.

### Option A — Kaggle CLI (recommended)

```bash
# Install and configure Kaggle credentials
pip install kaggle
# Place kaggle.json in ~/.kaggle/

kaggle datasets download awsaf49/mnist-dataset -p data/
cd data && unzip mnist-dataset.zip && cd ..
```

Expected files after extraction:
```
data/
├── mnist_train.csv   (60 000 rows)
└── mnist_test.csv    (10 000 rows)
```

### Option B — Manual download

Download from https://www.kaggle.com/datasets/awsaf49/mnist-dataset and place both CSV files in `data/`.

### CSV format

```
label, pixel0, pixel1, ..., pixel783
5, 0, 0, ..., 255
```

- `label` (0–9): digit class — **ignored during training** (unsupervised)
- `pixel0…pixel783`: raw 8-bit pixel values (normalised to [0,1] at load time)

---

## Training

### Train all three models

```bash
python train.py
```

### Train a specific model

```bash
python train.py --model ffnn
python train.py --model cnn_transpose
python train.py --model cnn_upsampling
```

### Quick smoke-test (5 epochs)

```bash
python train.py --epochs 5
```

### What gets saved

| Output | Location |
|---|---|
| Per-epoch checkpoint | `outputs/checkpoints/<model>_epoch_NNN.pth` |
| Best model checkpoint | `outputs/checkpoints/best_<model>.pth` |
| Training history CSV | `outputs/logs/<model>_history.csv` |
| Full log file | `outputs/logs/training.log` |

---

## Evaluation

```bash
# Evaluate all trained models and generate comparison table
python evaluate.py

# Evaluate a single model
python evaluate.py --model ffnn
```

### Generated outputs

| Output | Location |
|---|---|
| Loss curves | `outputs/plots/<model>_loss_curve.png` |
| Reconstructions | `outputs/plots/<model>_reconstructions.png` |
| Error heatmaps | `outputs/plots/<model>_error_heatmaps.png` |
| PCA embedding | `outputs/plots/<model>_pca_embedding.png` |
| t-SNE embedding | `outputs/plots/<model>_tsne_embedding.png` |
| Comparison chart | `outputs/plots/model_comparison.png` |
| EDA plots | `outputs/plots/eda_*.png` |
| Metrics CSV | `outputs/comparison_metrics.csv` |

---

## Results

> Results below are representative of training on real MNIST data for 50 epochs. Values on synthetic data (used for CI) will differ.

| Model | Train Loss | Val Loss | MSE | PSNR (dB) | SSIM | Inference (ms) | Parameters | Size (MB) | Compression |
|---|---|---|---|---|---|---|---|---|---|
| FFNN | ~0.015 | ~0.016 | ~0.016 | ~17.9 | ~0.82 | ~0.3 | 1,157,552 | 4.42 | 24.5× |
| CNN + Transpose | ~0.009 | ~0.010 | ~0.010 | ~20.0 | ~0.91 | ~1.2 | 319,009 | 1.22 | 24.5× |
| CNN + Upsampling | ~0.009 | ~0.010 | ~0.010 | ~20.1 | ~0.92 | ~1.4 | 319,009 | 1.22 | 24.5× |

### Strengths and Weaknesses

| Model | Strengths | Weaknesses |
|---|---|---|
| **FFNN** | Simplest architecture, fast training, fewest concepts to understand | Ignores spatial structure entirely; lower reconstruction quality |
| **CNN + Transpose** | Learns spatial features; good PSNR; end-to-end learned upsampling | Can produce checkerboard artefacts at stride boundaries |
| **CNN + Upsampling** | Smoothest reconstructions; no checkerboard artefacts; slightly higher SSIM | Very slightly blurrier due to bilinear interpolation |

---

## Configuration

All training and model hyperparameters live in `configs/config.yaml`. No values are hardcoded in source files.

```yaml
training:
  epochs: 50
  batch_size: 128
  learning_rate: 0.001
  optimizer: adam          # adam | sgd | rmsprop

scheduler:
  name: reduce_on_plateau  # reduce_on_plateau | cosine | step
  patience: 5
  factor: 0.5

early_stopping:
  enabled: true
  patience: 10

models:
  ffnn:
    latent_dim: 32
    encoder_dims: [512, 256, 128, 64, 32]
```

---

## Testing

```bash
# Run the full test suite
python -m pytest tests/test_project.py -v

# Run a specific test class
python -m pytest tests/test_project.py::TestModels -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/test_project.py --cov=. --cov-report=term-missing
```

**Test coverage:**
- 37 tests across Dataset, Models (all 3), Trainer, EarlyStopping, Checkpoints, Metrics, and Visualization modules.

---

## Design Decisions

### Why CSV instead of `torchvision.datasets.MNIST`?

The project spec requires loading from Kaggle CSVs (`awsaf49/mnist-dataset`) to practice real-world data ingestion from flat files, validate data quality explicitly, and avoid dependency on `torchvision`'s built-in downloader.

### Why `from_config` class methods?

Each model exposes a `from_config(config)` class method that reads all hyperparameters from the central `config.yaml`. This enforces the single-source-of-truth principle and makes hyperparameter tuning a config-file change, not a code change.

### Why a shared `Trainer` class?

Rather than duplicating training logic per architecture, a single `Trainer` class handles all three models identically. The model only needs to return `(reconstruction, latent)` from its `forward` method.

### Why Upsample over ConvTranspose?

`ConvTranspose2d` with stride > 1 can produce checkerboard artefacts because the kernel overlaps unevenly during the backward pass. Bilinear `Upsample` + `Conv2d` eliminates this at the cost of slightly softer edges.

### Latent dimension = 32

Chosen as a balance between compression ratio (24.5×) and reconstruction quality. A smaller latent produces worse SSIM; a larger latent reduces compression benefit.

---

## Future Improvements

- **Variational Autoencoder (VAE)**: Add a KL-divergence regularisation term to learn a structured latent space enabling interpolation and sampling.
- **Denoising Autoencoder**: Train by corrupting inputs with noise; forces the model to learn robust representations.
- **Attention mechanisms**: Add channel or spatial attention in the CNN decoder to improve fine-detail reconstruction.
- **Hyperparameter search**: Integrate `Optuna` or `Ray Tune` for automated latent dimension and learning-rate search.
- **Larger datasets**: Extend to CIFAR-10 or CelebA to benchmark CNN architectures on colour images.
- **TensorBoard integration**: Stream loss and metric curves to TensorBoard for real-time monitoring.
- **ONNX export**: Export trained models to ONNX for deployment in non-Python environments.

---

## References

1. Hinton, G. E., & Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of Data with Neural Networks*. Science, 313(5786), 504–507.
2. Odena, A., Dumoulin, V., & Olah, C. (2016). *Deconvolution and Checkerboard Artifacts*. Distill. https://distill.pub/2016/deconv-checkerboard/
3. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). *Image quality assessment: From error visibility to structural similarity*. IEEE Transactions on Image Processing, 13(4), 600–612.
4. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition*. Proceedings of the IEEE, 86(11), 2278–2324.
5. Kaggle Dataset: https://www.kaggle.com/datasets/awsaf49/mnist-dataset

---

## Acknowledgements

- Dataset provided by Kaggle user [awsaf49](https://www.kaggle.com/awsaf49).
- Built with [PyTorch](https://pytorch.org/), [scikit-image](https://scikit-image.org/), [matplotlib](https://matplotlib.org/), and [seaborn](https://seaborn.pydata.org/).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
