"""
mnist_dataset.py
----------------
Custom PyTorch Dataset and DataLoader factory for the MNIST CSV dataset.

Loads pixel data from ``mnist_train.csv`` and ``mnist_test.csv`` (from the
Kaggle dataset ``awsaf49/mnist-dataset``), validates the data, normalizes
pixel values to [0, 1], and exposes a standard Dataset interface.

Labels are carried internally for evaluation/visualization but are **never**
fed to any autoencoder during training (the task is unsupervised).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------

class MNISTCSVDataset(Dataset):
    """PyTorch Dataset wrapping the Kaggle MNIST CSV files.

    Each sample is a normalised ``(1, 28, 28)`` float tensor together with
    an integer class label.  The label is returned for evaluation purposes
    but is intentionally ignored by all autoencoder training loops.

    Args:
        csv_path: Path to a CSV file with columns ``label, pixel0, …, pixel783``.
        image_size: Side length of the square output image (default ``28``).

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If the CSV fails any of the data-integrity checks.
    """

    EXPECTED_PIXEL_COLS: int = 784
    PIXEL_PREFIX: str = "pixel"

    def __init__(self, csv_path: str, image_size: int = 28) -> None:
        self.csv_path = Path(csv_path)
        self.image_size = image_size

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        logger.info("Loading dataset from '%s'…", csv_path)
        raw_df = pd.read_csv(self.csv_path)

        self._validate(raw_df)

        pixel_cols = [f"{self.PIXEL_PREFIX}{i}" for i in range(self.EXPECTED_PIXEL_COLS)]
        self._labels: np.ndarray = raw_df["label"].values.astype(np.int64)
        raw_pixels: np.ndarray = raw_df[pixel_cols].values.astype(np.float32)

        # Normalise [0, 255] → [0, 1]
        self._pixels: np.ndarray = raw_pixels / 255.0

        logger.info(
            "Dataset ready — %d samples | image_size=%dx%d | "
            "pixel range [%.3f, %.3f]",
            len(self._pixels),
            image_size,
            image_size,
            float(self._pixels.min()),
            float(self._pixels.max()),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:
        """Run data-integrity checks and raise ``ValueError`` on failure.

        Checks performed:
        - Expected columns (``label`` + 784 pixel columns) are present.
        - No missing values.
        - No exact duplicate rows.
        - Pixel values are in the expected range [0, 255].
        - Each image flattens to exactly 784 values.

        Args:
            df: Raw dataframe loaded from the CSV.

        Raises:
            ValueError: If any check fails.
        """
        pixel_cols = [f"{self.PIXEL_PREFIX}{i}" for i in range(self.EXPECTED_PIXEL_COLS)]
        required_cols = {"label"} | set(pixel_cols)
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"CSV is missing columns: {missing_cols}")

        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            raise ValueError(f"Dataset contains {missing_count} missing values.")

        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            logger.warning(
                "%d duplicate rows found — they will be kept but flagged.",
                duplicate_count,
            )

        pixel_data = df[pixel_cols].values
        if pixel_data.min() < 0 or pixel_data.max() > 255:
            raise ValueError(
                f"Pixel values out of expected range [0, 255]: "
                f"found [{pixel_data.min()}, {pixel_data.max()}]."
            )

        if pixel_data.shape[1] != self.EXPECTED_PIXEL_COLS:
            raise ValueError(
                f"Expected {self.EXPECTED_PIXEL_COLS} pixel columns, "
                f"found {pixel_data.shape[1]}."
            )

        logger.info(
            "Data validation passed — %d rows, %d pixel columns, "
            "%d duplicate rows, 0 missing values.",
            len(df),
            self.EXPECTED_PIXEL_COLS,
            duplicate_count,
        )

    # ------------------------------------------------------------------
    # Dataset Protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._pixels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return the image tensor and label for sample at ``idx``.

        Args:
            idx: Integer index in ``[0, len(dataset))``.

        Returns:
            Tuple of:
            - ``image`` — float32 tensor of shape ``(1, 28, 28)``.
            - ``label`` — integer class label in ``[0, 9]``.
        """
        pixel_row = self._pixels[idx]  # (784,)
        image = torch.from_numpy(
            pixel_row.reshape(1, self.image_size, self.image_size)
        )
        label = int(self._labels[idx])
        return image, label

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self) -> Dict[str, object]:
        """Compute and return descriptive statistics for the dataset.

        Returns:
            Dictionary with keys: ``num_samples``, ``image_shape``,
            ``num_classes``, ``class_distribution``, ``pixel_mean``,
            ``pixel_std``, ``pixel_min``, ``pixel_max``.
        """
        class_counts = dict(
            zip(*np.unique(self._labels, return_counts=True))
        )
        stats = {
            "num_samples": len(self._pixels),
            "image_shape": (1, self.image_size, self.image_size),
            "num_classes": len(class_counts),
            "class_distribution": {int(k): int(v) for k, v in class_counts.items()},
            "pixel_mean": round(float(self._pixels.mean()), 6),
            "pixel_std": round(float(self._pixels.std()), 6),
            "pixel_min": round(float(self._pixels.min()), 6),
            "pixel_max": round(float(self._pixels.max()), 6),
        }
        return stats

    # ------------------------------------------------------------------
    # Accessors (used by visualization / evaluation)
    # ------------------------------------------------------------------

    @property
    def labels(self) -> np.ndarray:
        """Integer label array of shape ``(N,)``."""
        return self._labels

    @property
    def pixels(self) -> np.ndarray:
        """Normalised pixel array of shape ``(N, 784)``."""
        return self._pixels


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    config: Dict,
    train_csv_override: Optional[str] = None,
    test_csv_override: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test :class:`DataLoader` objects.

    Reads ``mnist_train.csv`` and ``mnist_test.csv`` paths from config,
    constructs :class:`MNISTCSVDataset` instances, splits the training set
    into train (48 000) and validation (12 000) subsets, and returns all
    three :class:`DataLoader` instances.

    Args:
        config: Full project configuration dictionary (from ``config.yaml``).
        train_csv_override: Optional override for the training CSV path.
        test_csv_override: Optional override for the test CSV path.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader)``.
    """
    paths = config["paths"]
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]

    train_csv = train_csv_override or paths["train_csv"]
    test_csv = test_csv_override or paths["test_csv"]
    image_size = dataset_cfg["image_size"]

    # ── Full training dataset ─────────────────────────────────────────────
    full_train_dataset = MNISTCSVDataset(train_csv, image_size=image_size)
    test_dataset = MNISTCSVDataset(test_csv, image_size=image_size)

    # ── Train / Validation split ──────────────────────────────────────────
    train_size = dataset_cfg["train_size"]
    val_size = dataset_cfg["val_size"]
    total = len(full_train_dataset)

    # Adjust if CSV has fewer rows than expected (e.g. during unit tests)
    if total < train_size + val_size:
        val_size = max(1, total // 5)
        train_size = total - val_size
        logger.warning(
            "Dataset smaller than expected (%d rows). "
            "Using train=%d, val=%d.",
            total,
            train_size,
            val_size,
        )

    generator = torch.Generator().manual_seed(training_cfg["random_seed"])
    train_subset, val_subset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    # ── Common DataLoader kwargs ──────────────────────────────────────────
    batch_size = training_cfg["batch_size"]
    num_workers = training_cfg.get("num_workers", 0)
    pin_memory = training_cfg.get("pin_memory", False)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    _log_split_info(full_train_dataset, test_dataset, train_size, val_size, batch_size)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _log_split_info(
    train_ds: MNISTCSVDataset,
    test_ds: MNISTCSVDataset,
    train_size: int,
    val_size: int,
    batch_size: int,
) -> None:
    """Log dataset statistics and split information.

    Args:
        train_ds: Full training :class:`MNISTCSVDataset`.
        test_ds: Test :class:`MNISTCSVDataset`.
        train_size: Number of samples in the training split.
        val_size: Number of samples in the validation split.
        batch_size: Mini-batch size used by the :class:`DataLoader`.
    """
    stats = train_ds.statistics()
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info("  Training samples    : %d", train_size)
    logger.info("  Validation samples  : %d", val_size)
    logger.info("  Test samples        : %d", len(test_ds))
    logger.info("  Batch size          : %d", batch_size)
    logger.info("  Image shape         : %s", stats["image_shape"])
    logger.info("  Pixel mean / std    : %.4f / %.4f", stats["pixel_mean"], stats["pixel_std"])
    logger.info("  Pixel range         : [%.3f, %.3f]", stats["pixel_min"], stats["pixel_max"])
    logger.info("  Classes             : %d", stats["num_classes"])
    logger.info("  Class distribution  : %s", stats["class_distribution"])
    logger.info("=" * 60)
