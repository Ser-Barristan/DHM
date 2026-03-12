# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
HoloYOLO — Gabor Stem Module
FILE PATH IN YOUR FORK: ultralytics/nn/modules/gabor.py
CREATE THIS FILE (it does not exist in vanilla ultralytics)

Gabor filter bank stem for holographic fringe detection.
Replaces the first Conv(ch, 16, 3, 2) layer in YOLOv8n backbone.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("GaborStem",)


class GaborFilterBank(nn.Module):
    """
    Learnable Gabor filter bank.

    Initialised to holographic fringe frequencies (40× objective,
    typical fringe spacing 8–20 px → normalised freq 0.05–0.20).
    All parameters remain trainable so the network can adapt.

    Args:
        in_channels  (int): input image channels (1 = grayscale, 3 = RGB)
        out_channels (int): output channels after 1×1 mixing conv
        kernel_size  (int): Gabor kernel spatial size (odd, ≥7)
        base_freqs  (tuple): initial spatial frequencies in cycles/pixel
        n_orient     (int): number of orientations uniformly spaced in [0, π)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        kernel_size: int = 15,
        base_freqs: tuple = (0.05, 0.10, 0.15, 0.20),
        n_orient: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        n_freq = len(base_freqs)
        n_gabor = n_freq * n_orient                        # 32 filters by default

        # ── Learnable filter parameters ──────────────────────────────────────
        # frequencies: repeat each freq for every orientation
        freq_init = torch.tensor(base_freqs, dtype=torch.float32).repeat(n_orient)
        self.frequencies = nn.Parameter(freq_init)          # (n_gabor,)

        # orientations: n_orient angles, each repeated n_freq times
        theta_init = torch.linspace(0.0, math.pi, n_orient + 1)[:-1]  # [0, π)
        theta_init = theta_init.repeat_interleave(n_freq)
        self.thetas = nn.Parameter(theta_init)              # (n_gabor,)

        # Gaussian envelope widths
        self.sigma_x = nn.Parameter(torch.full((n_gabor,), 3.0))
        self.sigma_y = nn.Parameter(torch.full((n_gabor,), 3.0))

        # Carrier phase offset
        self.psi = nn.Parameter(torch.zeros(n_gabor))

        self._n_gabor = n_gabor

        # ── 1×1 mixer: (n_gabor * in_channels) → out_channels ───────────────
        self.mix = nn.Conv2d(n_gabor * in_channels, out_channels, kernel_size=1, bias=False)
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_kernels(self) -> torch.Tensor:
        """Construct (n_gabor, 1, k, k) Gabor kernels from learnable params."""
        k    = self.kernel_size
        half = k // 2
        dev  = self.frequencies.device

        yy, xx = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32, device=dev),
            torch.arange(-half, half + 1, dtype=torch.float32, device=dev),
            indexing="ij",
        )                                                    # (k, k)

        kernels = []
        for i in range(self._n_gabor):
            cos_t = torch.cos(self.thetas[i])
            sin_t = torch.sin(self.thetas[i])
            sx    = self.sigma_x[i].abs().clamp(min=0.5)
            sy    = self.sigma_y[i].abs().clamp(min=0.5)

            x_rot =  xx * cos_t + yy * sin_t
            y_rot = -xx * sin_t + yy * cos_t

            gauss   = torch.exp(-0.5 * (x_rot ** 2 / sx ** 2 + y_rot ** 2 / sy ** 2))
            carrier = torch.cos(2.0 * math.pi * self.frequencies[i] * x_rot + self.psi[i])
            kernels.append(gauss * carrier)

        return torch.stack(kernels).unsqueeze(1)             # (n_gabor, 1, k, k)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, out_channels, H, W)  — same spatial size (padding=k//2)
        """
        kernels = self._build_kernels()                      # (n_gabor, 1, k, k)
        pad     = self.kernel_size // 2

        # Apply bank to every input channel independently, then concat
        channel_responses = []
        for c in range(self.in_channels):
            xc  = x[:, c : c + 1]                           # (B, 1, H, W)
            out = F.conv2d(xc, kernels, padding=pad)         # (B, n_gabor, H, W)
            channel_responses.append(out)

        fused = torch.cat(channel_responses, dim=1)          # (B, n_gabor*C, H, W)
        return self.act(self.bn(self.mix(fused)))             # (B, out_channels, H, W)


class GaborStem(nn.Module):
    """
    Drop-in replacement for YOLOv8n's first Conv block.

    Vanilla YOLOv8n backbone layer-0:
        - [-1, 1, Conv, [16, 3, 2]]   # Conv(ch, 16, k=3, s=2)

    Replace with:
        - [-1, 1, GaborStem, [16, 1]] # GaborStem(out_ch=16, in_ch=1)
        NOTE: yaml args order is [out_channels, in_channels]

    Architecture:
        GaborFilterBank(in_ch → 32)
        → DW-conv(32, k=3, s=2)       # stride-2 downsample
        → PW-conv(32 → out_ch)
        → BN → SiLU

    Parameter count: ~20 K  (vs ~450 for standard Conv stem — still tiny)

    Args:
        out_channels (int): number of output channels (matches vanilla layer-0 output)
        in_channels  (int): 1 for raw grayscale hologram, 3 for RGB
    """

    def __init__(self, out_channels: int = 16, in_channels: int = 1):
        super().__init__()
        out_channels = int(out_channels)   # yaml can deliver floats
        in_channels  = int(in_channels)
        mid = 32
        self.gabor = GaborFilterBank(
            in_channels=in_channels,
            out_channels=mid,
            kernel_size=15,
            base_freqs=(0.05, 0.10, 0.15, 0.20),
            n_orient=8,
        )
        # Stride-2 depthwise-separable downsample (cheap)
        self.dw  = nn.Conv2d(mid, mid, kernel_size=3, stride=2, padding=1, groups=mid, bias=False)
        self.pw  = nn.Conv2d(mid, out_channels, kernel_size=1, bias=False)
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gabor(x)           # (B, 32,      H,   W)
        x = self.dw(x)              # (B, 32,      H/2, W/2)
        x = self.act(self.bn(self.pw(x)))   # (B, out_ch, H/2, W/2)
        return x
