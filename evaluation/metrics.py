"""
metrics.py
----------
Reconstruction quality and model efficiency metrics.

Computes MSE, PSNR, SSIM, inference time, model size, parameter count,
and compression ratio for any trained autoencoder.

All functions operate on NumPy arrays (converted from torch tensors before
calling), and are written to be architecture-agnostic so they work with
FFNN, CNN-Transpose, and CNN-Upsampling autoencoders without modification.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim_fn

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Image Quality Metrics
# ---------------------------------------------------------------------------

def compute_mse(
    originals: np.ndarray,
    reconstructions: np.ndarray,
) -> float:
    """Compute mean squared error between originals and reconstructions.

    Args:
        originals: Ground-truth images of shape ``(N, H, W)`` or ``(N, C, H, W)``.
        reconstructions: Predicted images of the same shape.

    Returns:
        Scalar MSE value (float).
    """
    return float(np.mean((originals - reconstructions) ** 2))


def compute_psnr(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels.

    PSNR = 10 * log10(max_val^2 / MSE)

    Args:
        originals: Ground-truth images of shape ``(N, ...)``.
        reconstructions: Predicted images of the same shape.
        data_range: Maximum possible pixel value (``1.0`` for normalised images).

    Returns:
        Scalar PSNR value in dB (float).  Returns ``inf`` when MSE is zero.
    """
    mse = compute_mse(originals, reconstructions)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def compute_ssim(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Compute mean Structural Similarity Index (SSIM) over a batch.

    SSIM is computed per-image and averaged.  Both arrays are expected to
    be in ``(N, H, W)`` layout (single-channel images).

    Args:
        originals: Ground-truth images of shape ``(N, H, W)``.
        reconstructions: Predicted images of shape ``(N, H, W)``.
        data_range: Pixel value range (``1.0`` for normalised images).

    Returns:
        Mean SSIM value in ``[−1, 1]`` (float).
    """
    orig = originals.reshape(-1, originals.shape[-2], originals.shape[-1])
    recon = reconstructions.reshape(-1, reconstructions.shape[-2], reconstructions.shape[-1])

    scores = [
        ssim_fn(orig[i], recon[i], data_range=data_range)
        for i in range(len(orig))
    ]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Inference Time
# ---------------------------------------------------------------------------

def measure_inference_time(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    n_runs: int = 50,
) -> float:
    """Measure average inference latency over multiple forward passes.

    The first run is used as a warm-up and excluded from timing.

    Args:
        model: Trained autoencoder model.
        sample_input: Single-sample tensor of shape ``(1, C, H, W)``.
        device: Compute device to run inference on.
        n_runs: Number of forward passes to average (excluding warm-up).

    Returns:
        Mean inference time in **milliseconds** per sample.
    """
    model.eval()
    x = sample_input.to(device)

    # Warm-up
    with torch.no_grad():
        model(x)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(x)
    elapsed = time.perf_counter() - start

    mean_ms = (elapsed / n_runs) * 1000.0
    logger.debug("Inference time: %.4f ms / sample", mean_ms)
    return round(mean_ms, 4)


# ---------------------------------------------------------------------------
# Model Size & Compression
# ---------------------------------------------------------------------------

def compute_compression_ratio(
    input_dim: int,
    latent_dim: int,
) -> float:
    """Compute the spatial compression ratio of the bottleneck layer.

    Compression ratio = input_dim / latent_dim.

    Args:
        input_dim: Number of input features (e.g. ``784`` for 28×28 images).
        latent_dim: Bottleneck dimensionality.

    Returns:
        Compression ratio as a float.
    """
    return round(input_dim / latent_dim, 2)


# ---------------------------------------------------------------------------
# Full Metrics Bundle
# ---------------------------------------------------------------------------

def compute_all_metrics(
    model: nn.Module,
    originals: np.ndarray,
    reconstructions: np.ndarray,
    model_name: str,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute and return the complete set of evaluation metrics.

    Args:
        model: Trained autoencoder model (for size and timing measurements).
        originals: Ground-truth images of shape ``(N, 1, 28, 28)`` or ``(N, 784)``.
        reconstructions: Reconstructed images of the same shape.
        model_name: Human-readable model identifier (used in logging).
        device: Compute device.
        config: Full project configuration dictionary.

    Returns:
        Dictionary with keys:
        - ``model_name``
        - ``MSE``
        - ``PSNR``
        - ``SSIM``
        - ``Inference_Time_ms``
        - ``Parameters``
        - ``Size_MB``
        - ``Compression_Ratio``
    """
    from utils.helpers import count_parameters, model_size_mb

    # Reshape to (N, 28, 28)
    orig_2d = originals.reshape(-1, 28, 28)
    recon_2d = reconstructions.reshape(-1, 28, 28)

    mse = compute_mse(orig_2d, recon_2d)
    psnr = compute_psnr(orig_2d, recon_2d)
    ssim = compute_ssim(orig_2d, recon_2d)

    sample_input = torch.zeros(1, 1, 28, 28, device=device)
    inference_ms = measure_inference_time(model, sample_input, device)

    params = count_parameters(model)
    size = model_size_mb(model)
    latent_dim = config["models"]["ffnn"]["latent_dim"]
    compression = compute_compression_ratio(
        config["dataset"]["input_dim"], latent_dim
    )

    metrics = {
        "model_name": model_name,
        "MSE": round(mse, 8),
        "PSNR": round(psnr, 4),
        "SSIM": round(ssim, 6),
        "Inference_Time_ms": inference_ms,
        "Parameters": params,
        "Size_MB": round(size, 4),
        "Compression_Ratio": compression,
    }

    logger.info("=" * 60)
    logger.info("EVALUATION METRICS — %s", model_name)
    logger.info("=" * 60)
    for key, val in metrics.items():
        if key != "model_name":
            logger.info("  %-25s : %s", key, val)
    logger.info("=" * 60)

    return metrics
