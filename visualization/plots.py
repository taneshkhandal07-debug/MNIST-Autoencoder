"""
plots.py
--------
All plotting functions for the MNIST Autoencoder project.

Covers exploratory data analysis (EDA) plots, training history curves,
original-vs-reconstructed image grids, reconstruction error heatmaps,
and latent-space embeddings (PCA & t-SNE).

Every function saves its output to a configurable directory and returns the
full save path so callers can log or chain results.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.logger import get_logger

logger = get_logger(__name__)

# Consistent style across all figures
sns.set_theme(style="whitegrid", palette="muted")
_CMAP_ERROR = "hot"
_LABEL_FONT = {"fontsize": 12, "fontweight": "bold"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> str:
    """Save a matplotlib figure and close it.

    Args:
        fig: Figure to save.
        path: Destination file path (PNG).
        dpi: Resolution in dots per inch.

    Returns:
        The resolved save path string.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", path)
    return path


# ---------------------------------------------------------------------------
# EDA Plots
# ---------------------------------------------------------------------------

def plot_random_samples(
    pixels: np.ndarray,
    labels: np.ndarray,
    plots_dir: str,
    n_samples: int = 16,
    dpi: int = 150,
) -> str:
    """Display a grid of randomly selected MNIST images.

    Args:
        pixels: Normalised pixel array of shape ``(N, 784)``.
        labels: Integer label array of shape ``(N,)``.
        plots_dir: Directory to save the plot.
        n_samples: Number of images to show (should be a perfect square).
        dpi: Figure resolution.

    Returns:
        Save path of the generated PNG.
    """
    indices = np.random.choice(len(pixels), n_samples, replace=False)
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    fig.suptitle("Random MNIST Samples", **_LABEL_FONT, y=1.01)

    for ax_idx, sample_idx in enumerate(indices):
        row, col = divmod(ax_idx, cols)
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.imshow(pixels[sample_idx].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Label: {labels[sample_idx]}", fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for ax_idx in range(n_samples, rows * cols):
        row, col = divmod(ax_idx, cols)
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(plots_dir, "eda_random_samples.png")
    return _save_fig(fig, save_path, dpi)


def plot_class_distribution(
    labels: np.ndarray,
    plots_dir: str,
    dpi: int = 150,
) -> str:
    """Bar chart of sample count per digit class.

    Args:
        labels: Integer label array of shape ``(N,)``.
        plots_dir: Directory to save the plot.
        dpi: Figure resolution.

    Returns:
        Save path of the generated PNG.
    """
    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(unique, counts, color=sns.color_palette("muted", len(unique)), edgecolor="white")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Digit Class", **_LABEL_FONT)
    ax.set_ylabel("Sample Count", **_LABEL_FONT)
    ax.set_title("Class Distribution", **_LABEL_FONT)
    ax.set_xticks(unique)

    save_path = os.path.join(plots_dir, "eda_class_distribution.png")
    return _save_fig(fig, save_path, dpi)


def plot_pixel_histogram(
    pixels: np.ndarray,
    plots_dir: str,
    dpi: int = 150,
) -> str:
    """Histogram of normalised pixel intensity values across the dataset.

    Args:
        pixels: Normalised pixel array of shape ``(N, 784)``.
        plots_dir: Directory to save the plot.
        dpi: Figure resolution.

    Returns:
        Save path of the generated PNG.
    """
    flat = pixels.ravel()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(flat, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Pixel Intensity (normalised)", **_LABEL_FONT)
    ax.set_ylabel("Frequency", **_LABEL_FONT)
    ax.set_title("Pixel Intensity Distribution", **_LABEL_FONT)
    ax.axvline(float(flat.mean()), color="red", linestyle="--", label=f"Mean: {flat.mean():.3f}")
    ax.legend(fontsize=10)

    save_path = os.path.join(plots_dir, "eda_pixel_histogram.png")
    return _save_fig(fig, save_path, dpi)


def plot_sample_grid(
    pixels: np.ndarray,
    labels: np.ndarray,
    plots_dir: str,
    dpi: int = 150,
) -> str:
    """Grid of sample images, one per digit class.

    Args:
        pixels: Normalised pixel array of shape ``(N, 784)``.
        labels: Integer label array of shape ``(N,)``.
        plots_dir: Directory to save the plot.
        dpi: Figure resolution.

    Returns:
        Save path of the generated PNG.
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("One Sample per Class", **_LABEL_FONT)

    for digit in range(10):
        row, col = divmod(digit, 5)
        ax = axes[row, col]
        mask = labels == digit
        if mask.sum() > 0:
            idx = np.where(mask)[0][0]
            ax.imshow(pixels[idx].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Digit {digit}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(plots_dir, "eda_sample_grid.png")
    return _save_fig(fig, save_path, dpi)


# ---------------------------------------------------------------------------
# Training Curves
# ---------------------------------------------------------------------------

def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    model_name: str,
    plots_dir: str,
    dpi: int = 150,
) -> str:
    """Line chart of training and validation loss over epochs.

    Args:
        train_losses: List of per-epoch training loss values.
        val_losses: List of per-epoch validation loss values.
        model_name: Model identifier used in the plot title and filename.
        plots_dir: Directory to save the plot.
        dpi: Figure resolution.

    Returns:
        Save path of the generated PNG.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, label="Train Loss", color="#4C72B0", linewidth=2)
    ax.plot(epochs, val_losses, label="Val Loss", color="#DD8452", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", **_LABEL_FONT)
    ax.set_ylabel("Loss (MSE)", **_LABEL_FONT)
    ax.set_title(f"Training History — {model_name}", **_LABEL_FONT)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(plots_dir, f"{model_name}_loss_curve.png")
    return _save_fig(fig, save_path, dpi)


# ---------------------------------------------------------------------------
# Reconstruction Plots
# ---------------------------------------------------------------------------

def plot_reconstructions(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    model_name: str,
    plots_dir: str,
    n_samples: int = 10,
    dpi: int = 150,
) -> str:
    """Side-by-side grid of original and reconstructed MNIST images.

    Args:
        originals: Original image array of shape ``(N, 1, 28, 28)`` or ``(N, 784)``.
        reconstructions: Reconstructed array, same shape as ``originals``.
        model_name: Model identifier used in title and filename.
        plots_dir: Directory to save the plot.
        n_samples: Number of image pairs to display.
        dpi: Figure resolution.

    Returns:
        Save path of the generated PNG.
    """
    n_samples = min(n_samples, len(originals))
    orig = originals[:n_samples].reshape(-1, 28, 28)
    recon = reconstructions[:n_samples].reshape(-1, 28, 28)

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.5, 3.5))
    fig.suptitle(f"Original vs Reconstructed — {model_name}", **_LABEL_FONT)

    for i in range(n_samples):
        axes[0, i].imshow(orig[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10, rotation=90, labelpad=5)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10, rotation=90, labelpad=5)

    plt.tight_layout()
    save_path = os.path.join(plots_dir, f"{model_name}_reconstructions.png")
    return _save_fig(fig, save_path, dpi)


def plot_error_heatmaps(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    model_name: str,
    plots_dir: str,
    n_samples: int = 10,
    dpi: int = 150,
) -> str:
    """Three-row grid: original / reconstructed / absolute error heatmap.

    Args:
        originals: Original images of shape ``(N, 1, 28, 28)`` or ``(N, 784)``.
        reconstructions: Reconstructed images (same shape).
        model_name: Model identifier used in title and filename.
        plots_dir: Directory to save the plot.
        n_samples: Number of image triplets to display.
        dpi: Figure resolution.

    Returns:
        Save path of the generated PNG.
    """
    n_samples = min(n_samples, len(originals))
    orig = originals[:n_samples].reshape(-1, 28, 28)
    recon = reconstructions[:n_samples].reshape(-1, 28, 28)
    error = np.abs(orig - recon)

    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 1.5, 5))
    fig.suptitle(f"Reconstruction Error Heatmap — {model_name}", **_LABEL_FONT)

    row_labels = ["Original", "Reconstructed", "Abs Error"]
    cmaps = ["gray", "gray", _CMAP_ERROR]

    for row, (row_data, cmap, label) in enumerate(zip([orig, recon, error], cmaps, row_labels)):
        for col in range(n_samples):
            ax = axes[row, col]
            ax.imshow(row_data[col], cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=90, labelpad=4)

    plt.tight_layout()
    save_path = os.path.join(plots_dir, f"{model_name}_error_heatmaps.png")
    return _save_fig(fig, save_path, dpi)


# ---------------------------------------------------------------------------
# Latent Space Embeddings
# ---------------------------------------------------------------------------

def plot_pca_embedding(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    plots_dir: str,
    n_components: int = 2,
    dpi: int = 150,
) -> str:
    """2-D PCA scatter plot of the latent space coloured by class.

    Args:
        latent_vectors: Encoded representations of shape ``(N, latent_dim)``.
        labels: Integer class labels of shape ``(N,)``.
        model_name: Model identifier used in title and filename.
        plots_dir: Directory to save the plot.
        n_components: Number of PCA components (default ``2``).
        dpi: Figure resolution.

    Returns:
        Save path of the generated PNG.
    """
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(latent_vectors)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.5,
        s=10,
    )
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Digit Class", fontsize=10)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} var.)", **_LABEL_FONT)
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} var.)", **_LABEL_FONT)
    ax.set_title(f"PCA Latent Embedding — {model_name}", **_LABEL_FONT)

    save_path = os.path.join(plots_dir, f"{model_name}_pca_embedding.png")
    return _save_fig(fig, save_path, dpi)


def plot_tsne_embedding(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    plots_dir: str,
    perplexity: int = 30,
    n_iter: int = 1000,
    dpi: int = 150,
) -> str:
    """2-D t-SNE scatter plot of the latent space coloured by class.

    Args:
        latent_vectors: Encoded representations of shape ``(N, latent_dim)``.
        labels: Integer class labels of shape ``(N,)``.
        model_name: Model identifier used in title and filename.
        plots_dir: Directory to save the plot.
        perplexity: t-SNE perplexity parameter.
        n_iter: Number of t-SNE optimisation iterations.
        dpi: Figure resolution.

    Returns:
        Save path of the generated PNG.
    """
    logger.info("Computing t-SNE for %d samples (perplexity=%d)…", len(latent_vectors), perplexity)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=42,
        verbose=0,
    )
    reduced = tsne.fit_transform(latent_vectors)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.5,
        s=10,
    )
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Digit Class", fontsize=10)
    ax.set_xlabel("t-SNE Dim 1", **_LABEL_FONT)
    ax.set_ylabel("t-SNE Dim 2", **_LABEL_FONT)
    ax.set_title(f"t-SNE Latent Embedding — {model_name}", **_LABEL_FONT)

    save_path = os.path.join(plots_dir, f"{model_name}_tsne_embedding.png")
    return _save_fig(fig, save_path, dpi)


# ---------------------------------------------------------------------------
# Comparison Plot
# ---------------------------------------------------------------------------

def plot_model_comparison(
    metrics_df,
    plots_dir: str,
    dpi: int = 150,
) -> str:
    """Bar chart matrix comparing key metrics across all three models.

    Args:
        metrics_df: :class:`pandas.DataFrame` with model names as index and
            metric columns (``MSE``, ``PSNR``, ``SSIM``, ``Parameters``).
        plots_dir: Directory to save the plot.
        dpi: Figure resolution.

    Returns:
        Save path of the generated PNG.
    """
    metric_cols = [c for c in ["MSE", "PSNR", "SSIM", "Inference_Time_ms"] if c in metrics_df.columns]

    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = sns.color_palette("muted", len(metrics_df))
    models = metrics_df.index.tolist()

    for ax, metric in zip(axes, metric_cols):
        vals = metrics_df[metric].values
        bars = ax.bar(models, vals, color=colors, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(metric, **_LABEL_FONT)
        ax.set_ylabel(metric, fontsize=10)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Model Comparison", **_LABEL_FONT, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "model_comparison.png")
    return _save_fig(fig, save_path, dpi)
