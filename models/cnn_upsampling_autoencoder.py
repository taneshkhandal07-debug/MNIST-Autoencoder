"""
cnn_upsampling_autoencoder.py
------------------------------
Convolutional Autoencoder using ``Upsample`` + ``Conv2d`` in the decoder
instead of ``ConvTranspose2d``.

Motivation
~~~~~~~~~~
Transposed convolutions can produce checkerboard artefacts due to uneven
overlap in the kernel stride.  Replacing them with bilinear upsampling
followed by a standard convolution avoids this issue and often produces
smoother reconstructions.

Architecture (28×28 input → latent_dim vector → 28×28 output)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Encoder (identical to CNNTransposeAutoencoder):
  Conv2d(1,  32, k=3, s=2, p=1)  → 14×14
  Conv2d(32, 64, k=3, s=2, p=1)  → 7×7
  Conv2d(64,128, k=3, s=2, p=1)  → 4×4
  Flatten → Linear(128*4*4, latent_dim)

Decoder (Upsample + Conv):
  Linear(latent_dim, 128*4*4) → reshape (128, 4, 4)
  Upsample(scale=2) → 8×8  + Conv2d(128,64,k=3,p=1)
  Upsample(scale=2) → 16×16 + Conv2d(64, 32,k=3,p=1)
  Upsample(scale=2) → 32×32 + Conv2d(32,  1,k=3,p=1)  → centre-crop 28×28
  Sigmoid
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Encoder  (re-uses same structure as CNNTransposeAutoencoder)
# ---------------------------------------------------------------------------

class _CNNEncoder(nn.Module):
    """Convolutional encoder (shared architecture with CNNTranspose variant).

    Args:
        encoder_channels: Output channels per Conv2d block.
        latent_dim: Bottleneck vector size.
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
        self._flat_dim = encoder_channels[-1] * 4 * 4
        self.bottleneck = nn.Linear(self._flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        features = self.conv_layers(x)
        z = self.bottleneck(features.view(features.size(0), -1))
        return z, features.shape


# ---------------------------------------------------------------------------
# Decoder (Upsample + Conv)
# ---------------------------------------------------------------------------

class _UpsamplingDecoder(nn.Module):
    """Decoder that uses bilinear upsampling + Conv2d for reconstruction.

    Args:
        decoder_channels: Per-block output channel counts.
        latent_dim: Bottleneck dimensionality.
        flat_dim: Flattened feature size after projection.
        upsample_mode: Interpolation method (``'bilinear'`` or ``'nearest'``).
    """

    def __init__(
        self,
        decoder_channels: List[int],
        latent_dim: int,
        flat_dim: int,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self._upsample_mode = upsample_mode
        self.proj = nn.Linear(latent_dim, flat_dim)
        self._in_channels = flat_dim // 16  # 4×4 spatial

        # One Upsample + Conv block per decoder channel
        self.blocks = nn.ModuleList()
        in_ch = self._in_channels
        for i, out_ch in enumerate(decoder_channels):
            is_last = i == len(decoder_channels) - 1
            if is_last:
                block = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.Sigmoid(),
                )
            else:
                block = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            self.blocks.append(block)
            in_ch = out_ch

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to reconstructed images.

        Args:
            z: Latent tensor of shape ``(B, latent_dim)``.

        Returns:
            Reconstructed images of shape ``(B, 1, 28, 28)``.
        """
        x = self.proj(z)
        x = x.view(x.size(0), self._in_channels, 4, 4)
        for block in self.blocks:
            x = block(x)
        # Centre-crop in case upsampled output exceeds 28×28
        if x.shape[-1] != 28:
            x = x[:, :, :28, :28]
        return x


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class CNNUpsamplingAutoencoder(nn.Module):
    """CNN Autoencoder using Upsample + Conv2d for decoding.

    Produces smoother reconstructions than transposed convolutions by
    avoiding the checkerboard artefacts associated with uneven kernel
    overlap.

    Args:
        encoder_channels: Conv2d output channels for each encoder block.
        decoder_channels: Conv2d output channels for each decoder block.
        latent_dim: Bottleneck dimensionality.
        upsample_mode: Interpolation method for :class:`torch.nn.Upsample`.

    Example:
        >>> model = CNNUpsamplingAutoencoder()
        >>> x = torch.randn(4, 1, 28, 28)
        >>> recon, latent = model(x)
        >>> recon.shape   # (4, 1, 28, 28)
        >>> latent.shape  # (4, 32)
    """

    def __init__(
        self,
        encoder_channels: List[int] = None,
        decoder_channels: List[int] = None,
        latent_dim: int = 32,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [32, 64, 128]
        if decoder_channels is None:
            decoder_channels = [64, 32, 1]

        self.latent_dim = latent_dim

        self._encoder = _CNNEncoder(encoder_channels, latent_dim)
        flat_dim = encoder_channels[-1] * 4 * 4
        self._decoder = _UpsamplingDecoder(
            decoder_channels, latent_dim, flat_dim, upsample_mode
        )

        logger.info(
            "CNNUpsamplingAutoencoder created — latent_dim=%d, upsample_mode=%s",
            latent_dim,
            upsample_mode,
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
    def from_config(cls, config: Dict) -> "CNNUpsamplingAutoencoder":
        """Construct from the project configuration dictionary.

        Args:
            config: Full project configuration (loaded from ``config.yaml``).

        Returns:
            Initialised :class:`CNNUpsamplingAutoencoder`.
        """
        cfg = config["models"]["cnn_upsampling"]
        enc_ch = cfg["encoder_channels"][1:]   # exclude input channel [1]
        dec_ch = cfg["decoder_channels"][1:]   # exclude first channel [128]
        return cls(
            encoder_channels=enc_ch,
            decoder_channels=dec_ch,
            latent_dim=cfg["latent_dim"],
            upsample_mode=cfg.get("upsample_mode", "bilinear"),
        )
