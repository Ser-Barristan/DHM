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
        base_freqs: tuple = (0.05, 0.08, 0.12, 0.16, 0.20, 0.25),
        n_orient: int = 12,
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

    def _build_kernels(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Construct (n_gabor, 1, k, k) Gabor kernels from learnable params.

        Args:
            dtype:  match the input tensor dtype (float32 or float16 under AMP)
            device: match the input tensor device

        Note: all trig ops are performed in float32 for numerical stability,
        then cast to the requested dtype at the very end.  This avoids NaN
        in half-precision sin/cos while still satisfying F.conv2d type checks.
        """
        k    = self.kernel_size
        half = k // 2

        # Always build grid in float32 for precision
        yy, xx = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32, device=device),
            torch.arange(-half, half + 1, dtype=torch.float32, device=device),
            indexing="ij",
        )                                                    # (k, k)

        # Cast learnable params to float32 for the computation
        freqs   = self.frequencies.float()
        thetas  = self.thetas.float()
        sigma_x = self.sigma_x.float()
        sigma_y = self.sigma_y.float()
        psi     = self.psi.float()

        kernels = []
        for i in range(self._n_gabor):
            cos_t = torch.cos(thetas[i])
            sin_t = torch.sin(thetas[i])
            sx    = sigma_x[i].abs().clamp(min=0.5)
            sy    = sigma_y[i].abs().clamp(min=0.5)

            x_rot =  xx * cos_t + yy * sin_t
            y_rot = -xx * sin_t + yy * cos_t

            gauss   = torch.exp(-0.5 * (x_rot ** 2 / sx ** 2 + y_rot ** 2 / sy ** 2))
            carrier = torch.cos(2.0 * math.pi * freqs[i] * x_rot + psi[i])
            kernels.append(gauss * carrier)

        # Stack in float32 then cast to match input — fixes AMP HalfTensor mismatch
        return torch.stack(kernels).unsqueeze(1).to(dtype=dtype, device=device)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)  — float32 or float16 (AMP)
        Returns:
            out: (B, out_channels, H, W)  — same dtype as input
        """
        # Build kernels matching the input's dtype and device every forward pass.
        # This is cheap (no backward through the mesh) and fixes the AMP
        # HalfTensor / FloatTensor mismatch.
        kernels = self._build_kernels(dtype=x.dtype, device=x.device)
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

    Replace with (yaml):
        - [-1, 1, GaborStem, [16]]    # GaborStem(out_channels=16)
        NOTE: only ONE arg in yaml — out_channels.
              in_channels is read automatically from ch[] by tasks.py
              via the GaborStem branch you added to parse_model().

    Architecture:
        GaborFilterBank(in_ch → 64, same spatial size)
        → DW-conv(32, k=3, s=2)          # stride-2 downsample
        → PW-conv(32 → out_channels)
        → BN → SiLU

    Parameter count: ~20 K  (vs ~450 for standard Conv stem)

    Args:
        out_channels (int): output channel count (e.g. 16 for YOLOv8n)
        in_channels  (int): input channels — passed by parse_model via ch[f].
                            Defaults to 1 (grayscale hologram).
                            parse_model sets this automatically; you do NOT
                            put it in the yaml args list.
    """

    def __init__(self, out_channels: int = 16, in_channels: int = 1):
        super().__init__()
        out_channels = int(out_channels)
        in_channels  = int(in_channels)    # 1 = grayscale, 3 = RGB
        mid          = 64

        self.in_channels = in_channels     # stored so forward can validate

        self.gabor = GaborFilterBank(
            in_channels=in_channels,
            out_channels=mid,
            kernel_size=15,
            base_freqs=(0.05, 0.08, 0.12, 0.16, 0.20, 0.25),
            n_orient=12,
        )
        # Stride-2 depthwise-separable (cheap) downsample
        self.dw  = nn.Conv2d(mid, mid, kernel_size=3, stride=2, padding=1, groups=mid, bias=False)
        self.pw  = nn.Conv2d(mid, out_channels, kernel_size=1, bias=False)
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Safety: if input has more channels than expected (e.g. model built
        # with ch=3 but grayscale passed), use only the declared channels.
        # In practice parse_model guarantees this matches.
        if x.shape[1] != self.in_channels:
            x = x[:, : self.in_channels]      # trim silently
        x = self.gabor(x)                     # (B, 32,     H,   W)
        x = self.dw(x)                        # (B, 32,     H/2, W/2)
        x = self.act(self.bn(self.pw(x)))     # (B, out_ch, H/2, W/2)
        return x
