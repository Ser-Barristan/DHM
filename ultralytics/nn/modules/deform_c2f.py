# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
HoloYOLO — Deformable C2f for Neck
FILE PATH IN YOUR FORK: ultralytics/nn/modules/deform_c2f.py
CREATE THIS FILE (it does not exist in vanilla ultralytics)

DeformC2f replaces C2f in the neck PAN layers (P3 + P4 fusion).
Deformable conv offsets are predicted per-pixel so the network
learns to sample along CIRCULAR ring contours rather than rigid grids.

Requires:  torchvision >= 0.8  for deform_conv2d.
           Falls back to standard Conv2d with a warning if not available.

Fixes vs v1:
  - import math moved to TOP (was below class that used it → NameError)
  - ALL channel dims cast to int() — yaml parser delivers floats for n/c1/c2
    which caused:  TypeError: empty() received an invalid combination of arguments
"""

import math
import warnings

import torch
import torch.nn as nn

try:
    from torchvision.ops import deform_conv2d as _dcn
    _HAS_DCN = True
except ImportError:
    _HAS_DCN = False
    warnings.warn(
        "[HoloYOLO] torchvision.ops.deform_conv2d not found. "
        "DeformC2f will use standard Conv2d. "
        "Install torchvision ≥ 0.8 to enable deformable convolutions.",
        stacklevel=2,
    )


__all__ = ("DeformConv2d", "DeformBottleneck", "DeformC2f")


# ── DeformConv2d ───────────────────────────────────────────────────────────────

class DeformConv2d(nn.Module):
    """
    Lightweight DCNv2 wrapper with learned offsets + modulation masks.

    The offset/mask predictor is a small 3×3 conv with zero-init weights
    so training starts at the identity (no deformation) and learns
    ring-following offsets gradually.

    Args:
        in_ch       (int): input channels
        out_ch      (int): output channels
        kernel_size (int): convolution kernel size (default 3)
        stride      (int): stride (default 1)
        padding     (int): padding (default 1)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        # cast all dims to plain Python int — yaml can deliver floats
        in_ch       = int(in_ch)
        out_ch      = int(out_ch)
        kernel_size = int(kernel_size)
        stride      = int(stride)
        padding     = int(padding)

        self.stride  = stride
        self.padding = padding
        self.k       = kernel_size

        # Main convolution weight
        self.weight = nn.Parameter(
            torch.empty(out_ch, in_ch, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if _HAS_DCN:
            # Offset predictor outputs 2·k² offsets + k² modulation masks
            self.offset_conv = nn.Conv2d(
                in_ch,
                3 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            )
            # Zero-init → identity at training start (critical for stability)
            nn.init.zeros_(self.offset_conv.weight)
            nn.init.zeros_(self.offset_conv.bias)
        else:
            # Graceful fallback
            self.fallback = nn.Conv2d(
                in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _HAS_DCN:
            return self.fallback(x)

        n  = self.k * self.k
        raw    = self.offset_conv(x)
        offset = raw[:, : 2 * n]                             # (B, 2k², H, W)
        mask   = torch.sigmoid(raw[:, 2 * n :])              # (B, k², H, W)

        return _dcn(
            x,
            offset,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask,
        )


# ── DeformBottleneck ───────────────────────────────────────────────────────────

class DeformBottleneck(nn.Module):
    """
    Bottleneck with a deformable 3×3 conv in the middle.

    Same interface as ultralytics Bottleneck(c1, c2, shortcut, g, k, e).

    Structure:
        1×1 Conv → BN → SiLU
        DeformConv2d(3×3) → BN → SiLU
        1×1 Conv → BN → SiLU
        + optional shortcut

    Args:
        c1       (int):   input channels
        c2       (int):   output channels
        shortcut (bool):  add residual connection if c1==c2 (default True)
        g        (int):   groups for 3×3 (unused here, kept for API compat)
        k        (tuple): kernel sizes (unused, kept for API compat)
        e        (float): expansion ratio (default 0.5)
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple = (3, 3),
        e: float = 0.5,
    ):
        super().__init__()
        c1  = int(c1)
        c2  = int(c2)
        c_  = int(c2 * float(e))

        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )
        self.dcn = DeformConv2d(c_, c_, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c_)
        self.act2 = nn.SiLU(inplace=True)

        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.add = shortcut and (c1 == c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y = self.act2(self.bn2(self.dcn(y)))
        y = self.cv2(y)
        return x + y if self.add else y


# ── DeformC2f ─────────────────────────────────────────────────────────────────

class DeformC2f(nn.Module):
    """
    C2f with DeformBottleneck — drop-in replacement for ultralytics C2f.

    Interface matches C2f(c1, c2, n, shortcut, g, e) exactly so the
    yaml parser treats it identically to C2f.

    Used ONLY in the neck (P3 and P4 fusion layers) to keep param
    count low while adding circular receptive fields where it matters most.

    In holoYOLOv8n.yaml, neck layers 12 and 15:
        BEFORE: - [-1, 1, C2f,      [128, False]]
        AFTER:  - [-1, 1, DeformC2f, [128, False]]

    Args:
        c1       (int):   input channels
        c2       (int):   output channels
        n        (int):   number of bottleneck repeats (default 1)
        shortcut (bool):  bottleneck shortcut (default False in neck)
        g        (int):   groups (default 1)
        e        (float): channel expansion (default 0.5)
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        # cast ALL args to correct Python types — yaml parser passes floats
        c1 = int(c1)
        c2 = int(c2)
        n  = int(n)   # ← root cause of the TypeError: n was a float
        g  = int(g)
        e  = float(e)

        self.c = int(c2 * e)   # hidden channel count

        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(inplace=True),
        )
        # int() on the full expression guarantees no float reaches Conv2d
        self.cv2 = nn.Sequential(
            nn.Conv2d(int((2 + n) * self.c), c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.m = nn.ModuleList(
            DeformBottleneck(self.c, self.c, shortcut=shortcut, g=g, e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split cv1 output into 2 chunks along channel dim
        y = list(self.cv1(x).chunk(2, dim=1))
        # Pass through deformable bottlenecks
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))
