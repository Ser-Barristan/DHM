"""
FILE: ultralytics/nn/modules/gabor.py
ADD THIS AS A NEW FILE IN YOUR FORK

Gabor-based stem replacement for holographic fringe detection.
Replaces the standard Conv stem in YOLOv8n with physics-informed
filters tuned to holographic ring/fringe spatial frequencies.

Keeps params << 2.5M because filters are partially fixed (not all learned).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaborFilter(nn.Module):
    """
    Learnable Gabor filter bank.
    Frequencies and orientations are initialized from hologram physics
    but remain trainable — so the network can fine-tune them.

    For 40x holographic images, fringe spacing is typically 8-20px,
    so base_frequencies are set accordingly.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        kernel_size: int = 15,
        base_frequencies=(0.05, 0.10, 0.15, 0.20),
        n_orientations: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_orientations = n_orientations
        self.n_frequencies = len(base_frequencies)

        # Total fixed Gabor filters = n_freq * n_orient
        n_gabor = self.n_frequencies * n_orientations  # 32 by default

        # Learnable parameters for each filter
        self.frequencies = nn.Parameter(
            torch.tensor(base_frequencies).repeat(n_orientations)
        )  # shape: (n_gabor,)

        thetas = torch.linspace(0, math.pi, n_orientations, dtype=torch.float32)
        self.thetas = nn.Parameter(thetas.repeat_interleave(self.n_frequencies))

        self.sigma_x = nn.Parameter(torch.ones(n_gabor) * 3.0)
        self.sigma_y = nn.Parameter(torch.ones(n_gabor) * 3.0)
        self.psi = nn.Parameter(torch.zeros(n_gabor))

        # 1x1 conv to mix Gabor responses → out_channels
        self.mix = nn.Conv2d(n_gabor * in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

        self._n_gabor = n_gabor

    def _build_kernels(self):
        """Construct Gabor kernels on-the-fly from learnable params."""
        k = self.kernel_size
        half = k // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32, device=self.frequencies.device),
            torch.arange(-half, half + 1, dtype=torch.float32, device=self.frequencies.device),
            indexing="ij",
        )  # (k, k)

        kernels = []
        for i in range(self._n_gabor):
            theta = self.thetas[i]
            freq = self.frequencies[i]
            sx = self.sigma_x[i].abs().clamp(min=0.5)
            sy = self.sigma_y[i].abs().clamp(min=0.5)
            psi = self.psi[i]

            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            x_rot = x * cos_t + y * sin_t
            y_rot = -x * sin_t + y * cos_t

            gaussian = torch.exp(
                -0.5 * (x_rot**2 / sx**2 + y_rot**2 / sy**2)
            )
            carrier = torch.cos(2 * math.pi * freq * x_rot + psi)
            kernel = gaussian * carrier  # (k, k)
            kernels.append(kernel)

        # Stack → (n_gabor, 1, k, k)
        return torch.stack(kernels).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)  — C=1 for grayscale hologram, C=3 for RGB
        """
        kernels = self._build_kernels()  # (n_gabor, 1, k, k)

        # Apply to each input channel separately
        outs = []
        for c in range(self.in_channels):
            xc = x[:, c : c + 1, :, :]  # (B, 1, H, W)
            out = F.conv2d(
                xc, kernels, padding=self.kernel_size // 2
            )  # (B, n_gabor, H, W)
            outs.append(out)

        out = torch.cat(outs, dim=1)  # (B, n_gabor*C, H, W)
        out = self.mix(out)           # (B, out_channels, H, W)
        return self.act(self.bn(out))


class GaborStem(nn.Module):
    """
    Drop-in replacement for YOLOv8's first Conv block (stride-2 stem).

    Standard YOLOv8n stem:
        Conv(3, 16, 3, 2)  →  produces stride-2 feature map

    This stem:
        GaborFilter(in_ch, 32)  +  stride-2 depthwise conv  →  same spatial output
        Parameter count: ~12K vs ~450 for standard (still tiny, <0.1% of budget)

    Usage in yaml (see holoYOLO.yaml):
        - [[-1], 1, GaborStem, [16, 1]]   # args: [out_ch, in_ch]
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 16):
        super().__init__()
        # Gabor bank → 32 channels
        self.gabor = GaborFilter(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=15,
            base_frequencies=(0.05, 0.10, 0.15, 0.20),
            n_orientations=8,
        )
        # Stride-2 depthwise-separable to downsample + compress
        self.dw = nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False)
        self.pw = nn.Conv2d(32, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gabor(x)
        x = self.pw(self.dw(x))
        return self.act(self.bn(x))
