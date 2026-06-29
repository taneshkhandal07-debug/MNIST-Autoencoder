"""
ffnn_autoencoder.py
-------------------
Feed-Forward (fully-connected) Autoencoder for MNIST image reconstruction.

Architecture
~~~~~~~~~~~~
Encoder  : 784 → 512 → 256 → 128 → 64 → 32 (latent)
Decoder  : 32 → 64 → 128 → 256 → 512 → 784

Each hidden layer uses ReLU activation with BatchNorm for training
stability.  The output layer uses Sigmoid to constrain pixel values to
[0, 1].  Weights are initialised from a zero-mean Normal distribution;
biases are initialised to zero.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class _EncoderBlock(nn.Module):
    """Single fully-connected encoder block: Linear → BatchNorm → ReLU.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder block.

        Args:
            x: Input tensor of shape ``(batch, in_features)``.

        Returns:
            Output tensor of shape ``(batch, out_features)``.
        """
        return self.block(x)


class _DecoderBlock(nn.Module):
    """Single fully-connected decoder block: Linear → BatchNorm → ReLU.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder block.

        Args:
            x: Input tensor of shape ``(batch, in_features)``.

        Returns:
            Output tensor of shape ``(batch, out_features)``.
        """
        return self.block(x)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class FFNNAutoencoder(nn.Module):
    """Feed-Forward Autoencoder for MNIST image reconstruction.

    The model flattens input images (``1×28×28``) to a 784-dimensional
    vector, compresses them through a symmetric encoder–decoder, and
    reshapes the output back to the original image dimensions.

    Args:
        input_dim: Flattened image size (default ``784``).
        encoder_dims: List of hidden-layer widths for the encoder.
        latent_dim: Bottleneck (code) dimension.
        decoder_dims: List of hidden-layer widths for the decoder.

    Example:
        >>> model = FFNNAutoencoder()
        >>> x = torch.randn(8, 1, 28, 28)
        >>> recon, latent = model(x)
        >>> recon.shape     # (8, 1, 28, 28)
        >>> latent.shape    # (8, 32)
    """

    def __init__(
        self,
        input_dim: int = 784,
        encoder_dims: List[int] = None,
        latent_dim: int = 32,
        decoder_dims: List[int] = None,
    ) -> None:
        super().__init__()

        if encoder_dims is None:
            encoder_dims = [512, 256, 128, 64]
        if decoder_dims is None:
            decoder_dims = [64, 128, 256, 512]

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ── Encoder ───────────────────────────────────────────────────────
        enc_layers: List[nn.Module] = []
        prev_dim = input_dim
        for dim in encoder_dims:
            enc_layers.append(_EncoderBlock(prev_dim, dim))
            prev_dim = dim
        enc_layers.append(nn.Linear(prev_dim, latent_dim))   # bottleneck (no activation)
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder ───────────────────────────────────────────────────────
        dec_layers: List[nn.Module] = []
        prev_dim = latent_dim
        for dim in decoder_dims:
            dec_layers.append(_DecoderBlock(prev_dim, dim))
            prev_dim = dim
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        dec_layers.append(nn.Sigmoid())                        # constrain to [0, 1]
        self.decoder = nn.Sequential(*dec_layers)

        # ── Weight initialisation ─────────────────────────────────────────
        self._init_weights()

        logger.info(
            "FFNNAutoencoder created — input_dim=%d, latent_dim=%d",
            input_dim,
            latent_dim,
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, std: float = 0.01) -> None:
        """Initialise Linear weights with N(0, std) and biases to zero.

        Args:
            std: Standard deviation for the normal distribution.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images to latent vectors.

        Args:
            x: Image tensor of shape ``(batch, 1, 28, 28)`` or
               ``(batch, 784)``.

        Returns:
            Latent tensor of shape ``(batch, latent_dim)``.
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.encoder(x_flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors back to image tensors.

        Args:
            z: Latent tensor of shape ``(batch, latent_dim)``.

        Returns:
            Reconstructed image tensor of shape ``(batch, 1, 28, 28)``.
        """
        out_flat = self.decoder(z)
        return out_flat.view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode then decode a batch of images.

        Args:
            x: Image tensor of shape ``(batch, 1, 28, 28)``.

        Returns:
            Tuple of:
            - ``reconstruction`` — tensor of shape ``(batch, 1, 28, 28)``.
            - ``latent`` — encoded tensor of shape ``(batch, latent_dim)``.
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent

    # ------------------------------------------------------------------
    # Class Method: from_config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict) -> "FFNNAutoencoder":
        """Construct a model from the project configuration dictionary.

        Args:
            config: Full project configuration (loaded from ``config.yaml``).

        Returns:
            An initialised :class:`FFNNAutoencoder` instance.
        """
        ffnn_cfg = config["models"]["ffnn"]
        dataset_cfg = config["dataset"]
        return cls(
            input_dim=dataset_cfg["input_dim"],
            encoder_dims=ffnn_cfg["encoder_dims"][:-1],   # last element is latent_dim
            latent_dim=ffnn_cfg["latent_dim"],
            decoder_dims=ffnn_cfg["decoder_dims"],
        )
