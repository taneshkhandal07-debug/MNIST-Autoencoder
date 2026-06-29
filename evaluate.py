"""
evaluate.py
-----------
Entry-point for evaluating trained MNIST autoencoder checkpoints.

Loads the best saved checkpoint for each model, runs inference on the
test set, computes all metrics, generates all plots, and prints a
side-by-side comparison table.

Usage
~~~~~
    python evaluate.py                          # evaluate all models
    python evaluate.py --model ffnn             # evaluate one model
    python evaluate.py --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from datasets.mnist_dataset import create_dataloaders
from evaluation.evaluator import Evaluator
from models.cnn_transpose_autoencoder import CNNTransposeAutoencoder
from models.cnn_upsampling_autoencoder import CNNUpsamplingAutoencoder
from models.ffnn_autoencoder import FFNNAutoencoder
from utils.helpers import (
    ensure_directories,
    get_device,
    load_checkpoint,
    load_config,
    set_seed,
)
from utils.logger import get_logger
from visualization.plots import plot_model_comparison

logger = get_logger(__name__)

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
        description="Evaluate MNIST Autoencoder checkpoints."
    )
    parser.add_argument(
        "--model",
        choices=list(_MODEL_REGISTRY.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_best_model(
    model_key: str,
    config: Dict,
    device: torch.device,
) -> Optional[torch.nn.Module]:
    """Load the best checkpoint for a given model key.

    Args:
        model_key: One of ``'ffnn'``, ``'cnn_transpose'``, ``'cnn_upsampling'``.
        config: Full project configuration dictionary.
        device: Compute device.

    Returns:
        Loaded model, or ``None`` if no checkpoint is found.
    """
    model_name = config["models"][model_key]["name"]
    ckpt_dir = Path(config["paths"]["checkpoints_dir"])
    best_path = ckpt_dir / f"best_{model_name}.pth"

    if not best_path.exists():
        logger.warning(
            "No best checkpoint found for '%s' at '%s'. "
            "Run train.py first.",
            model_key,
            best_path,
        )
        return None

    ModelClass = _MODEL_REGISTRY[model_key]
    model = ModelClass.from_config(config)
    load_checkpoint(str(best_path), model, device=device)
    return model


def _print_comparison_table(all_metrics: List[Dict[str, Any]]) -> None:
    """Print a formatted Markdown comparison table to stdout.

    Args:
        all_metrics: List of metric dictionaries, one per model.
    """
    if not all_metrics:
        return

    df = pd.DataFrame(all_metrics).set_index("model_name")
    numeric_cols = df.select_dtypes("number").columns

    logger.info("\n")
    logger.info("=" * 80)
    logger.info("MODEL COMPARISON TABLE")
    logger.info("=" * 80)
    print(df[numeric_cols].to_markdown())
    logger.info("=" * 80)

    # Also save as CSV
    df.to_csv("outputs/comparison_metrics.csv")
    logger.info("Comparison table saved → outputs/comparison_metrics.csv")

    # Strengths and weaknesses summary
    print("\n## Strengths and Weaknesses\n")
    print(
        "| Model             | Strengths                                          | Weaknesses                                    |"
    )
    print(
        "|-------------------|----------------------------------------------------|-----------------------------------------------|"
    )
    print(
        "| FFNN              | Fast training, simple architecture, low params     | Ignores spatial structure, lower SSIM          |"
    )
    print(
        "| CNN + Transpose   | Learns spatial hierarchies, good PSNR              | Checkerboard artefacts possible at boundaries  |"
    )
    print(
        "| CNN + Upsampling  | Smooth reconstructions, no checkerboard artefacts  | Slightly blurrier output due to interpolation  |"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Evaluate one or all models and produce comparison outputs."""
    args = _parse_args()
    config = load_config(args.config)

    set_seed(config["training"]["random_seed"])
    device = get_device(config)
    ensure_directories(config)

    _, _, test_loader = create_dataloaders(config)

    model_keys: List[str] = (
        list(_MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    )

    all_metrics: List[Dict[str, Any]] = []

    for key in model_keys:
        model = _load_best_model(key, config, device)
        if model is None:
            continue

        evaluator = Evaluator(
            model=model,
            model_name=config["models"][key]["name"],
            config=config,
            device=device,
        )
        metrics = evaluator.evaluate(test_loader)
        all_metrics.append(metrics)

    if len(all_metrics) > 1:
        _print_comparison_table(all_metrics)

        # Comparison bar chart
        import pandas as pd

        df = pd.DataFrame(all_metrics).set_index("model_name")
        plot_model_comparison(df, config["paths"]["plots_dir"])

    logger.info("Evaluation pipeline complete.")


if __name__ == "__main__":
    main()
