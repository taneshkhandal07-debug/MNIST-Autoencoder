"""
test_project.py
---------------
Comprehensive test suite for the MNIST Autoencoder project.

Covers:
- Dataset loading and validation
- Forward passes for all three architectures
- Training loop (2-epoch smoke test)
- Checkpoint save / load
- Evaluation metrics
- Visualization functions

Run with:
    python -m pytest tests/test_project.py -v
or directly:
    python tests/test_project.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.mnist_dataset import MNISTCSVDataset, create_dataloaders
from evaluation.metrics import (
    compute_compression_ratio,
    compute_mse,
    compute_psnr,
    compute_ssim,
    measure_inference_time,
)
from models.cnn_transpose_autoencoder import CNNTransposeAutoencoder
from models.cnn_upsampling_autoencoder import CNNUpsamplingAutoencoder
from models.ffnn_autoencoder import FFNNAutoencoder
from training.trainer import EarlyStopping, Trainer
from utils.helpers import (
    count_parameters,
    get_device,
    load_checkpoint,
    load_config,
    model_size_mb,
    save_checkpoint,
    set_seed,
)
from visualization.plots import (
    plot_class_distribution,
    plot_pixel_histogram,
    plot_random_samples,
    plot_reconstructions,
    plot_error_heatmaps,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def config():
    return load_config("configs/config.yaml")


@pytest.fixture(scope="session")
def device(config):
    return get_device(config)


@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory):
    """Create a tiny synthetic MNIST CSV for fast test execution."""
    import pandas as pd
    tmp = tmp_path_factory.mktemp("data")
    n = 200
    labels = np.random.randint(0, 10, n)
    pixels = np.random.randint(0, 256, (n, 784))
    cols = ["label"] + [f"pixel{i}" for i in range(784)]
    df = pd.DataFrame(np.column_stack([labels, pixels]), columns=cols)
    path = str(tmp / "test_mnist.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def sample_images():
    """Return a small batch of random images."""
    return torch.rand(8, 1, 28, 28)


# ---------------------------------------------------------------------------
# Dataset Tests
# ---------------------------------------------------------------------------

class TestDataset:
    def test_csv_loads(self, sample_csv):
        ds = MNISTCSVDataset(sample_csv)
        assert len(ds) == 200

    def test_image_shape(self, sample_csv):
        ds = MNISTCSVDataset(sample_csv)
        img, label = ds[0]
        assert img.shape == (1, 28, 28), f"Expected (1,28,28), got {img.shape}"

    def test_pixel_range(self, sample_csv):
        ds = MNISTCSVDataset(sample_csv)
        img, _ = ds[0]
        assert img.min() >= 0.0 and img.max() <= 1.0, "Pixels not normalised to [0,1]"

    def test_label_dtype(self, sample_csv):
        ds = MNISTCSVDataset(sample_csv)
        _, label = ds[0]
        assert isinstance(label, int)

    def test_statistics_keys(self, sample_csv):
        ds = MNISTCSVDataset(sample_csv)
        stats = ds.statistics()
        required = {"num_samples", "image_shape", "num_classes",
                    "class_distribution", "pixel_mean", "pixel_std"}
        assert required.issubset(set(stats.keys()))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            MNISTCSVDataset("nonexistent.csv")

    def test_dataloader_creation(self, config, sample_csv):
        train, val, test = create_dataloaders(
            config,
            train_csv_override=sample_csv,
            test_csv_override=sample_csv,
        )
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_batch_shape(self, config, sample_csv):
        train, _, _ = create_dataloaders(
            config,
            train_csv_override=sample_csv,
            test_csv_override=sample_csv,
        )
        images, labels = next(iter(train))
        assert images.ndim == 4
        assert images.shape[1:] == (1, 28, 28)


# ---------------------------------------------------------------------------
# Model Forward Pass Tests
# ---------------------------------------------------------------------------

class TestModels:
    @pytest.mark.parametrize("ModelClass", [
        FFNNAutoencoder,
        CNNTransposeAutoencoder,
        CNNUpsamplingAutoencoder,
    ])
    def test_forward_output_shape(self, config, sample_images, ModelClass):
        model = ModelClass.from_config(config)
        model.eval()
        with torch.no_grad():
            recon, latent = model(sample_images)
        assert recon.shape == sample_images.shape, \
            f"{ModelClass.__name__}: recon shape {recon.shape} ≠ {sample_images.shape}"
        assert latent.shape[0] == sample_images.shape[0]

    @pytest.mark.parametrize("ModelClass", [
        FFNNAutoencoder,
        CNNTransposeAutoencoder,
        CNNUpsamplingAutoencoder,
    ])
    def test_output_range(self, config, sample_images, ModelClass):
        """Sigmoid output must be in [0, 1]."""
        model = ModelClass.from_config(config)
        model.eval()
        with torch.no_grad():
            recon, _ = model(sample_images)
        assert recon.min().item() >= 0.0
        assert recon.max().item() <= 1.0

    @pytest.mark.parametrize("ModelClass", [
        FFNNAutoencoder,
        CNNTransposeAutoencoder,
        CNNUpsamplingAutoencoder,
    ])
    def test_latent_dim(self, config, sample_images, ModelClass):
        model = ModelClass.from_config(config)
        with torch.no_grad():
            _, latent = model(sample_images)
        expected_latent = config["models"]["ffnn"]["latent_dim"]
        assert latent.shape[1] == expected_latent

    def test_ffnn_parameter_count(self, config):
        model = FFNNAutoencoder.from_config(config)
        params = count_parameters(model)
        assert params > 0

    def test_model_size_positive(self, config):
        model = FFNNAutoencoder.from_config(config)
        size = model_size_mb(model)
        assert size > 0.0


# ---------------------------------------------------------------------------
# Training Tests
# ---------------------------------------------------------------------------

class TestTrainer:
    def test_training_reduces_loss(self, config, device, sample_csv):
        """Two-epoch training should not raise and loss should be finite."""
        from torch.utils.data import DataLoader
        from datasets.mnist_dataset import MNISTCSVDataset

        ds = MNISTCSVDataset(sample_csv)
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        model = FFNNAutoencoder.from_config(config)
        trainer = Trainer(model, config, "ffnn_test", device)
        history = trainer.train(loader, loader, epochs=2)

        assert len(history["train_loss"]) == 2
        assert all(np.isfinite(l) for l in history["train_loss"])

    def test_history_keys(self, config, device, sample_csv):
        from torch.utils.data import DataLoader
        from datasets.mnist_dataset import MNISTCSVDataset

        ds = MNISTCSVDataset(sample_csv)
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        model = FFNNAutoencoder.from_config(config)
        trainer = Trainer(model, config, "ffnn_keys_test", device)
        history = trainer.train(loader, loader, epochs=1)

        assert "train_loss" in history
        assert "val_loss" in history
        assert "lr" in history

    def test_early_stopping_triggers(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        triggered = False
        for _ in range(4):   # first call sets best; next 3 count as non-improving
            triggered = es.step(0.5)
        assert triggered

    def test_early_stopping_resets_on_improvement(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        es.step(0.5)
        es.step(0.4)   # improvement → counter resets
        es.step(0.4)
        es.step(0.4)
        assert not es.should_stop   # only 2 non-improving after reset


# ---------------------------------------------------------------------------
# Checkpoint Tests
# ---------------------------------------------------------------------------

class TestCheckpoints:
    def test_save_and_load(self, config, device, tmp_path):
        model = FFNNAutoencoder.from_config(config)
        optimizer = torch.optim.Adam(model.parameters())

        # Temporarily redirect checkpoint dir
        config_copy = dict(config)
        config_copy["paths"] = dict(config["paths"])
        config_copy["paths"]["checkpoints_dir"] = str(tmp_path)

        save_checkpoint(model, optimizer, epoch=1, loss=0.05,
                        config=config_copy, model_name="ffnn_ckpt_test", is_best=True)

        best_path = tmp_path / "best_ffnn_ckpt_test.pth"
        assert best_path.exists()

        model2 = FFNNAutoencoder.from_config(config)
        ckpt = load_checkpoint(str(best_path), model2, device=device)
        assert ckpt["epoch"] == 1
        assert abs(ckpt["loss"] - 0.05) < 1e-6

    def test_load_nonexistent_raises(self, device):
        model = FFNNAutoencoder()
        with pytest.raises(FileNotFoundError):
            load_checkpoint("no_such_file.pth", model, device=device)


# ---------------------------------------------------------------------------
# Evaluation Metrics Tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_mse_zero_for_identical(self):
        arr = np.random.rand(10, 28, 28).astype(np.float32)
        assert compute_mse(arr, arr) == pytest.approx(0.0, abs=1e-8)

    def test_mse_positive_for_different(self):
        a = np.ones((10, 28, 28), dtype=np.float32)
        b = np.zeros((10, 28, 28), dtype=np.float32)
        assert compute_mse(a, b) > 0.0

    def test_psnr_inf_for_identical(self):
        arr = np.random.rand(10, 28, 28).astype(np.float32)
        assert compute_psnr(arr, arr) == float("inf")

    def test_psnr_finite_for_different(self):
        a = np.ones((10, 28, 28), dtype=np.float32) * 0.8
        b = np.ones((10, 28, 28), dtype=np.float32) * 0.2
        psnr = compute_psnr(a, b)
        assert np.isfinite(psnr)

    def test_ssim_range(self):
        a = np.random.rand(5, 28, 28).astype(np.float32)
        b = np.random.rand(5, 28, 28).astype(np.float32)
        ssim = compute_ssim(a, b)
        assert -1.0 <= ssim <= 1.0

    def test_compression_ratio(self):
        ratio = compute_compression_ratio(784, 32)
        assert ratio == pytest.approx(24.5, rel=0.01)

    def test_inference_time_positive(self, config, device):
        model = FFNNAutoencoder.from_config(config).to(device)
        x = torch.zeros(1, 1, 28, 28)
        t = measure_inference_time(model, x, device, n_runs=5)
        assert t > 0.0


# ---------------------------------------------------------------------------
# Visualization Tests
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_random_samples_plot(self, tmp_path):
        pixels = np.random.rand(100, 784).astype(np.float32)
        labels = np.random.randint(0, 10, 100)
        path = plot_random_samples(pixels, labels, str(tmp_path), n_samples=4)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_class_distribution_plot(self, tmp_path):
        labels = np.array([i % 10 for i in range(100)])
        path = plot_class_distribution(labels, str(tmp_path))
        assert Path(path).exists()

    def test_pixel_histogram_plot(self, tmp_path):
        pixels = np.random.rand(100, 784).astype(np.float32)
        path = plot_pixel_histogram(pixels, str(tmp_path))
        assert Path(path).exists()

    def test_reconstruction_plot(self, tmp_path):
        orig = np.random.rand(10, 1, 28, 28).astype(np.float32)
        recon = np.random.rand(10, 1, 28, 28).astype(np.float32)
        path = plot_reconstructions(orig, recon, "test_model", str(tmp_path), n_samples=5)
        assert Path(path).exists()

    def test_error_heatmap_plot(self, tmp_path):
        orig = np.random.rand(10, 1, 28, 28).astype(np.float32)
        recon = np.random.rand(10, 1, 28, 28).astype(np.float32)
        path = plot_error_heatmaps(orig, recon, "test_model", str(tmp_path), n_samples=5)
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Runner (when called directly, not via pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
