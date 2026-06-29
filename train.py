"""
train.py
--------
Main entry-point for training one or all MNIST autoencoder architectures.

Usage
~~~~~
# Train all three models sequentially (default):
    python train.py

# Train a single model:
    python train.py --model ffnn
    python train.py --model cnn_transpose
    python train.py --model cnn_upsampling

# Override epochs for a quick smoke-test:
    python train.py --epochs 5

# Use a custom config file:
    python train.py --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is on sys.path when invoked from any directory
sys.path.insert(0, str(Path(__file__).parent))

from datasets.mnist_dataset import create_dataloaders
from models.cnn_transpose_autoencoder import CNNTransposeAutoencoder
from models.cnn_upsampling_autoencoder import CNNUpsamplingAutoencoder
from models.ffnn_autoencoder import FFNNAutoencoder
from training.trainer import Trainer
from utils.helpers import (
    ensure_directories,
    get_device,
    load_config,
    set_seed,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Model Registry ────────────────────────────────────────────────────────
_MODEL_REGISTRY = {
    "ffnn": FFNNAutoencoder,
    "cnn_transpose": CNNTransposeAutoencoder,
    "cnn_upsampling": CNNUpsamplingAutoencoder,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MNIST Autoencoder architectures."
    )
    parser.add_argument(
        "--model",
        choices=list(_MODEL_REGISTRY.keys()) + ["all"],
        default="all",
        help="Which model to train (default: all).",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config.yaml (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs from config.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_model(
    model_key: str,
    config: Dict,
    device,
    epochs: int,
) -> Dict:
    """Instantiate, train, and evaluate a single autoencoder.

    Args:
        model_key: One of ``'ffnn'``, ``'cnn_transpose'``, ``'cnn_upsampling'``.
        config: Full project configuration dictionary.
        device: Torch compute device.
        epochs: Number of training epochs.

    Returns:
        Training history dictionary from :class:`Trainer`.
    """
    logger.info("=" * 70)
    logger.info("TRAINING  —  %s", model_key.upper())
    logger.info("=" * 70)

    ModelClass = _MODEL_REGISTRY[model_key]
    model = ModelClass.from_config(config)

    train_loader, val_loader, _ = create_dataloaders(config)

    trainer = Trainer(
        model=model,
        config=config,
        model_name=config["models"][model_key]["name"],
        device=device,
    )

    history = trainer.train(train_loader, val_loader, epochs=epochs)
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry-point: parse args, run training for selected model(s)."""
    args = _parse_args()
    config = load_config(args.config)

    set_seed(config["training"]["random_seed"])
    device = get_device(config)
    ensure_directories(config)

    epochs = args.epochs or config["training"]["epochs"]

    if args.model == "all":
        model_keys: List[str] = list(_MODEL_REGISTRY.keys())
    else:
        model_keys = [args.model]

    for key in model_keys:
        _train_model(key, config, device, epochs)

    logger.info("All training runs complete.")


if __name__ == "__main__":
    main()
