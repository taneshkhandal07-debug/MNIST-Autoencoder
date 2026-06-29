"""
helpers.py
----------
Shared utility functions for the MNIST Autoencoder project.

Covers configuration loading, reproducibility seeding, device detection,
model persistence, and general-purpose file/metric helpers.
"""

import csv
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load the project YAML configuration file into a nested dictionary.

    Args:
        config_path: Relative or absolute path to ``config.yaml``.

    Returns:
        Dictionary containing all configuration key-value pairs.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    logger.info("Configuration loaded from '%s'", config_path)
    return config


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy, and PyTorch for reproducibility.

    Also configures cuDNN to use deterministic algorithms when a GPU is
    available.

    Args:
        seed: Integer seed value (default ``42``).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------

def get_device(config: Dict[str, Any]) -> torch.device:
    """Detect and return the best available compute device.

    Respects the ``device`` section of ``config.yaml``.  When
    ``auto_detect`` is ``true``, prefers CUDA > MPS > CPU.

    Args:
        config: Full project configuration dictionary.

    Returns:
        A :class:`torch.device` instance pointing to the selected backend.
    """
    device_cfg = config.get("device", {})
    auto_detect = device_cfg.get("auto_detect", True)

    if auto_detect:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        preferred = device_cfg.get("preferred", "cpu")
        device = torch.device(preferred)

    logger.info("Using device: %s", device)
    return device


# ---------------------------------------------------------------------------
# Directory Helpers
# ---------------------------------------------------------------------------

def ensure_directories(config: Dict[str, Any]) -> None:
    """Create all output directories defined in ``config.yaml`` if absent.

    Args:
        config: Full project configuration dictionary.
    """
    paths_cfg = config.get("paths", {})
    dirs = [
        paths_cfg.get("checkpoints_dir", "outputs/checkpoints"),
        paths_cfg.get("reconstructions_dir", "outputs/reconstructions"),
        paths_cfg.get("plots_dir", "outputs/plots"),
        paths_cfg.get("logs_dir", "outputs/logs"),
    ]
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug("Directory ready: %s", directory)


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: Dict[str, Any],
    model_name: str,
    is_best: bool = False,
) -> str:
    """Serialize a training checkpoint to disk.

    Saves both a per-epoch checkpoint and, if ``is_best`` is ``True``, a
    copy named ``best_<model_name>.pth``.

    Args:
        model: The PyTorch model whose state is being saved.
        optimizer: The optimizer whose state is being saved.
        epoch: Current training epoch (1-indexed).
        loss: Validation loss at this epoch.
        config: Full project configuration dictionary.
        model_name: Identifier string (e.g. ``"ffnn_autoencoder"``).
        is_best: If ``True``, also save as the best model checkpoint.

    Returns:
        Path string of the saved checkpoint file.
    """
    checkpoints_dir = config["paths"]["checkpoints_dir"]
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_name": model_name,
    }

    filename = os.path.join(checkpoints_dir, f"{model_name}_epoch_{epoch:03d}.pth")
    torch.save(checkpoint, filename)

    if is_best:
        best_path = os.path.join(checkpoints_dir, f"best_{model_name}.pth")
        torch.save(checkpoint, best_path)
        logger.info("Best model checkpoint saved: %s", best_path)

    return filename


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a model (and optionally optimizer) state from a checkpoint file.

    Args:
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        model: Model instance whose weights will be restored.
        optimizer: Optional optimizer to restore state into.
        device: Target device for loading tensors.

    Returns:
        The full checkpoint dictionary (includes epoch, loss, etc.).

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    map_location = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(
        "Checkpoint loaded: %s (epoch %d, loss %.6f)",
        checkpoint_path,
        checkpoint["epoch"],
        checkpoint["loss"],
    )
    return checkpoint


# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------

def save_metrics_to_csv(
    metrics: Dict[str, Any],
    csv_path: str,
) -> None:
    """Append a dictionary of metric values as a new row in a CSV file.

    Creates the file with a header row on first write.

    Args:
        metrics: Dictionary mapping metric names to values.
        csv_path: Destination CSV file path.
    """
    path = Path(csv_path)
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    logger.info("Metrics saved to %s", csv_path)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for measuring elapsed wall-clock time.

    Example:
        >>> with Timer("inference") as t:
        ...     model(x)
        >>> print(t.elapsed)
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            logger.debug("%s took %.4f seconds", self.label, self.elapsed)


# ---------------------------------------------------------------------------
# Model Info
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters in a model.

    Args:
        model: Any :class:`torch.nn.Module` subclass.

    Returns:
        Integer count of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size in megabytes based on parameter memory footprint.

    Args:
        model: Any :class:`torch.nn.Module` subclass.

    Returns:
        Approximate model size in MB (float).
    """
    num_params = count_parameters(model)
    size_bytes = num_params * 4  # float32 = 4 bytes
    return size_bytes / (1024 ** 2)


def get_model_info(model: torch.nn.Module, model_name: str) -> Dict[str, Union[str, int, float]]:
    """Collect a summary of model metadata.

    Args:
        model: The PyTorch model to inspect.
        model_name: Human-readable name for the model.

    Returns:
        Dictionary with keys: ``model_name``, ``parameters``, ``size_mb``.
    """
    params = count_parameters(model)
    size = model_size_mb(model)
    info = {
        "model_name": model_name,
        "parameters": params,
        "size_mb": round(size, 4),
    }
    logger.info(
        "Model '%s' — Parameters: %s | Size: %.4f MB",
        model_name,
        f"{params:,}",
        size,
    )
    return info
