"""
cnn_transpose_autoencoder.py
----------------------------
Convolutional Autoencoder using ``Conv2d`` for encoding and
``ConvTranspose2d`` for decoding.

Architecture (28×28 input → 7×7 latent spatial map → 28×28 output)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Encoder:
  Conv2d(1,  32, k=3, s=2, p=1)  → 14×14
  Conv2d(32, 64, k=3, s=2, p=1)  → 7×7
  Conv2d(64,128, k=3, s=2, p=1)  → 4×4
  Flatten → Linear(128*4*4, latent_dim)

Decoder:
  Linear(latent_dim, 128*4*4) → reshape (128, 4, 4)
  ConvTranspose2d(128,64, k=3, s=2, p=1, op=1) → 8×8
  ConvTranspose2d(64, 32, k=3, s=2, p=1, op=1) → 16×16
  ConvTranspose2d(32,  1, k=3, s=2, p=1, op=1) → 32×32 → centre-crop to 28×28
  Sigmoid

All hidden layers use BatchNorm2d + ReLU.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class _CNNEncoder(nn.Module):
    """Convolutional encoder that reduces 1×28×28 to a latent vector.

    Args:
        encoder_channels: List of output channel counts per Conv layer.
        latent_dim: Size of the bottleneck vector.
    """

    def __init__(self, encoder_channels: List[int], latent_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in encoder_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)

        # Determine the spatial size after convolutions (28→14→7→4 with stride=2)
        self._flat_dim = encoder_channels[-1] * 4 * 4
        self.bottleneck = nn.Linear(self._flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        """Encode input images.

        Args:
            x: Tensor of shape ``(B, 1, 28, 28)``.

        Returns:
            Tuple of ``(latent, conv_shape)`` where ``conv_shape`` is the
            shape before flattening (needed for the decoder).
        """
        features = self.conv_layers(x)
        conv_shape = features.shape
        z = self.bottleneck(features.view(features.size(0), -1))
        return z, conv_shape


# ---------------------------------------------------------------------------
# Decoder (Transpose)
# ---------------------------------------------------------------------------

class _CNNTransposeDecoder(nn.Module):
    """Decoder that reconstructs images via ``ConvTranspose2d``.

    Args:
        decoder_channels: List of output channels per transpose-conv layer.
        latent_dim: Bottleneck dimensionality.
        flat_dim: Total features expected after the projection linear layer.
    """

    def __init__(
        self,
        decoder_channels: List[int],
        latent_dim: int,
        flat_dim: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, flat_dim)
        self._in_channels = flat_dim // 16   # spatial size is 4×4

        layers: List[nn.Module] = []
        in_ch = self._in_channels
        for i, out_ch in enumerate(decoder_channels):
            is_last = i == len(decoder_channels) - 1
            layers.append(
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            if is_last:
                layers.append(nn.Sigmoid())
            else:
                layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            in_ch = out_ch

        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors back to images.

        Args:
            z: Latent tensor of shape ``(B, latent_dim)``.

        Returns:
            Reconstructed images of shape ``(B, 1, 28, 28)``.
        """
        x = self.proj(z)
        x = x.view(x.size(0), self._in_channels, 4, 4)
        x = self.deconv_layers(x)
        # Centre-crop to exactly 28×28 in case output is slightly larger
        if x.shape[-1] != 28:
            x = x[:, :, :28, :28]
        return x


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class CNNTransposeAutoencoder(nn.Module):
    """CNN Autoencoder using ConvTranspose2d for upsampling.

    Args:
        encoder_channels: Conv2d output channels for the encoder.
        decoder_channels: ConvTranspose2d output channels for the decoder.
        latent_dim: Bottleneck dimensionality.

    Example:
        >>> model = CNNTransposeAutoencoder()
        >>> x = torch.randn(4, 1, 28, 28)
        >>> recon, latent = model(x)
        >>> recon.shape     # (4, 1, 28, 28)
        >>> latent.shape    # (4, 32)
    """

    def __init__(
        self,
        encoder_channels: List[int] = None,
        decoder_channels: List[int] = None,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [32, 64, 128]
        if decoder_channels is None:
            decoder_channels = [64, 32, 1]

        self.latent_dim = latent_dim

        self._encoder = _CNNEncoder(encoder_channels, latent_dim)
        flat_dim = encoder_channels[-1] * 4 * 4
        self._decoder = _CNNTransposeDecoder(decoder_channels, latent_dim, flat_dim)

        logger.info(
            "CNNTransposeAutoencoder created — latent_dim=%d",
            latent_dim,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent vectors.

        Args:
            x: Image tensor of shape ``(B, 1, 28, 28)``.

        Returns:
            Latent tensor of shape ``(B, latent_dim)``.
        """
        z, _ = self._encoder(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to images.

        Args:
            z: Latent tensor of shape ``(B, latent_dim)``.

        Returns:
            Reconstructed images of shape ``(B, 1, 28, 28)``.
        """
        return self._decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode then decode.

        Args:
            x: Image tensor of shape ``(B, 1, 28, 28)``.

        Returns:
            Tuple of ``(reconstruction, latent)``.
        """
        z, _ = self._encoder(x)
        reconstruction = self._decoder(z)
        return reconstruction, z

    @classmethod
    def from_config(cls, config: Dict) -> "CNNTransposeAutoencoder":
        """Construct from the project configuration dictionary.

        Args:
            config: Full project configuration (loaded from ``config.yaml``).

        Returns:
            Initialised :class:`CNNTransposeAutoencoder`.
        """
        cfg = config["models"]["cnn_transpose"]
        # encoder_channels excludes the input channel (1) at index 0
        enc_ch = cfg["encoder_channels"][1:]   # [32, 64, 128]
        dec_ch = cfg["decoder_channels"][1:]   # [64, 32, 1]
        return cls(
            encoder_channels=enc_ch,
            decoder_channels=dec_ch,
            latent_dim=cfg["latent_dim"],
        )
