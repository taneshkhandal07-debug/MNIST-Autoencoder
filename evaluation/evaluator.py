"""
evaluator.py
------------
High-level evaluation pipeline for a trained autoencoder.

Runs inference on the test DataLoader, collects reconstructions, computes
all metrics, generates visualization plots (reconstructions, error heatmaps,
latent embeddings), and saves results to CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import compute_all_metrics
from utils.helpers import save_metrics_to_csv
from utils.logger import get_logger
from visualization.plots import (
    plot_error_heatmaps,
    plot_pca_embedding,
    plot_reconstructions,
    plot_tsne_embedding,
    plot_training_history,
)

logger = get_logger(__name__)


class Evaluator:
    """Orchestrates the end-to-end evaluation of a trained autoencoder.

    Args:
        model: Trained autoencoder (must expose ``encode`` and ``forward``).
        model_name: Identifier string used in filenames and log messages.
        config: Full project configuration dictionary.
        device: Compute device to run inference on.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.model_name = model_name
        self.config = config
        self.device = device
        self.plots_dir = config["paths"]["plots_dir"]
        self.eval_cfg = config.get("evaluation", {})

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _collect_outputs(
        self, loader: DataLoader, max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run inference over a DataLoader and collect all outputs.

        Args:
            loader: DataLoader yielding ``(images, labels)`` batches.
            max_samples: If set, stop after collecting this many samples.

        Returns:
            Tuple of:
            - ``originals``       — shape ``(N, 1, 28, 28)``
            - ``reconstructions`` — shape ``(N, 1, 28, 28)``
            - ``latent_vectors``  — shape ``(N, latent_dim)``
            - ``labels``          — shape ``(N,)``
        """
        self.model.eval()
        all_orig: List[np.ndarray] = []
        all_recon: List[np.ndarray] = []
        all_latent: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        n_collected = 0

        with torch.no_grad():
            for images, labels in loader:
                if max_samples and n_collected >= max_samples:
                    break

                images = images.to(self.device)
                reconstruction, latent = self.model(images)

                all_orig.append(images.cpu().numpy())
                all_recon.append(reconstruction.cpu().numpy())
                all_latent.append(latent.cpu().numpy())
                all_labels.append(labels.numpy())
                n_collected += len(images)

        originals = np.concatenate(all_orig, axis=0)
        reconstructions = np.concatenate(all_recon, axis=0)
        latent_vectors = np.concatenate(all_latent, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        logger.info(
            "Collected %d samples for evaluation (%s)",
            len(originals),
            self.model_name,
        )
        return originals, reconstructions, latent_vectors, labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_loader: DataLoader,
        history: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Any]:
        """Run full evaluation and generate all outputs.

        Steps performed:
        1. Collect reconstructions and latent vectors over ``test_loader``.
        2. Compute MSE, PSNR, SSIM, inference time, model size.
        3. Save metrics to the shared comparison CSV.
        4. Plot training history (if ``history`` is provided).
        5. Plot original vs reconstructed images.
        6. Plot error heatmaps.
        7. Plot PCA latent embedding.
        8. Plot t-SNE latent embedding (on a subsample for speed).

        Args:
            test_loader: DataLoader for the test split.
            history: Training history dict (from :class:`Trainer`).

        Returns:
            Metrics dictionary produced by :func:`compute_all_metrics`.
        """
        logger.info("Starting evaluation — model=%s", self.model_name)

        # ── 1. Inference ──────────────────────────────────────────────
        originals, reconstructions, latent_vectors, labels = self._collect_outputs(
            test_loader
        )

        # ── 2. Metrics ────────────────────────────────────────────────
        metrics = compute_all_metrics(
            model=self.model,
            originals=originals,
            reconstructions=reconstructions,
            model_name=self.model_name,
            device=self.device,
            config=self.config,
        )

        # ── 3. Save metrics CSV ───────────────────────────────────────
        metrics_csv = self.config["evaluation"].get(
            "metrics_csv", "outputs/comparison_metrics.csv"
        )
        save_metrics_to_csv(metrics, metrics_csv)

        # ── 4. Training history plot ──────────────────────────────────
        if history:
            plot_training_history(
                train_losses=history["train_loss"],
                val_losses=history["val_loss"],
                model_name=self.model_name,
                plots_dir=self.plots_dir,
            )

        # ── 5. Reconstruction grid ────────────────────────────────────
        n_samples = self.eval_cfg.get("num_reconstruction_samples", 10)
        plot_reconstructions(
            originals=originals,
            reconstructions=reconstructions,
            model_name=self.model_name,
            plots_dir=self.plots_dir,
            n_samples=n_samples,
        )

        # ── 6. Error heatmaps ─────────────────────────────────────────
        plot_error_heatmaps(
            originals=originals,
            reconstructions=reconstructions,
            model_name=self.model_name,
            plots_dir=self.plots_dir,
            n_samples=n_samples,
        )

        # ── 7. PCA embedding ──────────────────────────────────────────
        perplexity = self.eval_cfg.get("tsne_perplexity", 30)
        pca_n = self.eval_cfg.get("pca_components", 2)
        plot_pca_embedding(
            latent_vectors=latent_vectors,
            labels=labels,
            model_name=self.model_name,
            plots_dir=self.plots_dir,
            n_components=pca_n,
        )

        # ── 8. t-SNE embedding (subsample for performance) ────────────
        tsne_max = min(2000, len(latent_vectors))
        idx = np.random.choice(len(latent_vectors), tsne_max, replace=False)
        plot_tsne_embedding(
            latent_vectors=latent_vectors[idx],
            labels=labels[idx],
            model_name=self.model_name,
            plots_dir=self.plots_dir,
            perplexity=self.eval_cfg.get("tsne_perplexity", 30),
            n_iter=self.eval_cfg.get("tsne_n_iter", 1000),
        )

        logger.info("Evaluation complete — model=%s", self.model_name)
        return metrics
