"""
trainer.py
----------
Reusable training framework for all three MNIST autoencoder architectures.

Features
~~~~~~~~
- Training and validation loops with progress bars (tqdm)
- Best-model checkpoint saving
- Learning-rate scheduling (ReduceLROnPlateau / CosineAnnealingLR / StepLR)
- Early stopping with configurable patience and minimum delta
- Gradient clipping
- Per-epoch CSV history logging
- Complete training history returned as a dictionary
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.helpers import save_checkpoint
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Early Stopping Helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Monitor a validation metric and signal when training should halt.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change in the monitored metric to qualify as
            an improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss: float = float("inf")
        self._counter: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float) -> bool:
        """Update state with the latest validation loss.

        Args:
            val_loss: Current epoch validation loss.

        Returns:
            ``True`` if training should be stopped.
        """
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self.patience,
                )
        return self.should_stop


# ---------------------------------------------------------------------------
# Optimizer & Scheduler Factories
# ---------------------------------------------------------------------------

def _build_optimizer(
    model: nn.Module,
    config: Dict,
) -> torch.optim.Optimizer:
    """Instantiate an optimizer from ``config.yaml`` settings.

    Args:
        model: Model whose parameters will be optimised.
        config: Full project configuration dictionary.

    Returns:
        Configured :class:`torch.optim.Optimizer`.

    Raises:
        ValueError: If an unsupported optimizer name is specified.
    """
    train_cfg = config["training"]
    name = train_cfg["optimizer"].lower()
    lr = train_cfg["learning_rate"]
    wd = train_cfg.get("weight_decay", 0.0)

    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    if name == "rmsprop":
        return RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: '{name}'. Choose from adam, sgd, rmsprop.")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict,
) -> Optional[object]:
    """Instantiate a learning-rate scheduler from ``config.yaml`` settings.

    Args:
        optimizer: The optimizer to wrap.
        config: Full project configuration dictionary.

    Returns:
        A configured scheduler, or ``None`` if no scheduler is specified.
    """
    sched_cfg = config.get("scheduler", {})
    name = sched_cfg.get("name", "reduce_on_plateau").lower()

    if name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=sched_cfg.get("patience", 5),
            factor=sched_cfg.get("factor", 0.5),
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
            eta_min=sched_cfg.get("min_lr", 1e-6),
        )
    if name == "step":
        return StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 10),
            gamma=sched_cfg.get("gamma", 0.1),
        )
    logger.warning("Unknown scheduler '%s' — no scheduler will be used.", name)
    return None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Unified training driver for any autoencoder model.

    Handles the full training lifecycle: forward pass, loss computation,
    back-propagation, gradient clipping, validation, learning-rate
    scheduling, early stopping, checkpoint saving, and history logging.

    Args:
        model: Any :class:`torch.nn.Module` that returns
               ``(reconstruction, latent)`` from its ``forward`` method.
        config: Full project configuration dictionary.
        model_name: Unique identifier string (used for checkpoint filenames).
        device: Compute device (``cuda``, ``mps``, or ``cpu``).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        model_name: str,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.model_name = model_name
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = _build_optimizer(model, config)
        self.scheduler = _build_scheduler(self.optimizer, config)

        es_cfg = config.get("early_stopping", {})
        self.early_stopping: Optional[EarlyStopping] = (
            EarlyStopping(
                patience=es_cfg.get("patience", 10),
                min_delta=es_cfg.get("min_delta", 1e-5),
            )
            if es_cfg.get("enabled", True)
            else None
        )

        gc_cfg = config.get("gradient_clipping", {})
        self.grad_clip_enabled: bool = gc_cfg.get("enabled", True)
        self.grad_clip_max_norm: float = gc_cfg.get("max_norm", 1.0)

        self._best_val_loss: float = float("inf")
        self._history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        # CSV history log
        logs_dir = config["paths"]["logs_dir"]
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        self._csv_path = os.path.join(logs_dir, f"{model_name}_history.csv")
        self._init_csv()

        logger.info(
            "Trainer ready — model=%s | optimizer=%s | device=%s",
            model_name,
            config["training"]["optimizer"],
            device,
        )

    # ------------------------------------------------------------------
    # CSV Logging
    # ------------------------------------------------------------------

    def _init_csv(self) -> None:
        """Create the CSV file and write the header row."""
        with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    def _log_epoch_csv(self, epoch: int, train_loss: float, val_loss: float, lr: float) -> None:
        """Append one epoch's metrics to the CSV log.

        Args:
            epoch: Current epoch number (1-indexed).
            train_loss: Mean training loss for this epoch.
            val_loss: Mean validation loss for this epoch.
            lr: Current learning rate.
        """
        with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{lr:.8e}"])

    # ------------------------------------------------------------------
    # Single-epoch loops
    # ------------------------------------------------------------------

    def _train_one_epoch(self, loader: DataLoader) -> float:
        """Run one complete pass over the training DataLoader.

        Args:
            loader: Training :class:`DataLoader`.

        Returns:
            Mean training loss (MSE) for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc="  Train", leave=False, ncols=90)
        for images, _ in pbar:           # labels ignored — unsupervised task
            images = images.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            reconstruction, _ = self.model(images)
            loss = self.criterion(reconstruction, images)
            loss.backward()

            if self.grad_clip_enabled:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip_max_norm
                )

            self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            pbar.set_postfix({"loss": f"{batch_loss:.6f}"})

        return total_loss / max(num_batches, 1)

    def _validate_one_epoch(self, loader: DataLoader) -> float:
        """Run one complete pass over the validation DataLoader.

        Args:
            loader: Validation :class:`DataLoader`.

        Returns:
            Mean validation loss (MSE) for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(loader, desc="  Val  ", leave=False, ncols=90)
            for images, _ in pbar:
                images = images.to(self.device, non_blocking=True)
                reconstruction, _ = self.model(images)
                loss = self.criterion(reconstruction, images)
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Full Training Loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Execute the full training and validation loop.

        Args:
            train_loader: DataLoader for the training split.
            val_loader: DataLoader for the validation split.
            epochs: Number of epochs to train (overrides config if provided).

        Returns:
            Training history dictionary with keys ``train_loss``,
            ``val_loss``, and ``lr``.
        """
        num_epochs = epochs or self.config["training"]["epochs"]
        logger.info(
            "Starting training — model=%s | epochs=%d",
            self.model_name,
            num_epochs,
        )

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._validate_one_epoch(val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── History recording ─────────────────────────────────────
            self._history["train_loss"].append(train_loss)
            self._history["val_loss"].append(val_loss)
            self._history["lr"].append(current_lr)
            self._log_epoch_csv(epoch, train_loss, val_loss, current_lr)

            # ── Logging ───────────────────────────────────────────────
            logger.info(
                "Epoch [%03d/%03d] | train_loss=%.6f | val_loss=%.6f | lr=%.2e",
                epoch,
                num_epochs,
                train_loss,
                val_loss,
                current_lr,
            )

            # ── Scheduler step ────────────────────────────────────────
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # ── Checkpoint ────────────────────────────────────────────
            is_best = val_loss < self._best_val_loss
            if is_best:
                self._best_val_loss = val_loss

            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=val_loss,
                config=self.config,
                model_name=self.model_name,
                is_best=is_best,
            )

            # ── Early stopping ────────────────────────────────────────
            if self.early_stopping is not None:
                if self.early_stopping.step(val_loss):
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

        logger.info(
            "Training complete — best val_loss=%.6f",
            self._best_val_loss,
        )
        return self._history

    # ------------------------------------------------------------------
    # History Accessor
    # ------------------------------------------------------------------

    @property
    def history(self) -> Dict[str, List[float]]:
        """Training history with lists of per-epoch loss and LR values."""
        return self._history
