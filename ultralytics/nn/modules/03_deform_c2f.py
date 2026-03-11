"""
FILE: ultralytics/nn/modules/deform_c2f.py
ADD THIS AS A NEW FILE IN YOUR FORK

DeformC2f — C2f block with Deformable Convolution v2 bottlenecks.
Replaces standard C2f in the neck (P3/P4 feature fusion layers).

Why: Standard 3×3 conv assumes features are grid-aligned.
Holographic rings are CIRCULAR — their energy is NOT on a grid.
Deformable convs learn to sample along ring contours.

Uses lightweight DCNv2 (torchvision built-in) to stay under 2.5M params.
Only the neck C2f layers are replaced (not backbone) to keep param count low.

Estimated param overhead vs standard C2f(256,256,1): ~+15K per block
"""

import torch
import torch.nn as nn

try:
    from torchvision.ops import deform_conv2d
    HAS_DCN = True
except ImportError:
    HAS_DCN = False
    import warnings
    warnings.warn(
        "torchvision.ops.deform_conv2d not available — "
        "DeformBottleneck will fall back to standard Conv2d. "
        "Install torchvision >= 0.8 for deformable convolutions."
    )


class DeformConv2d(nn.Module):
    """
    Lightweight DCNv2 wrapper.
    Offset + mask are predicted from the input feature map by a small conv.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

        # Main weight
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_uniform_(self.weight)

        if HAS_DCN:
            # Offset predictor: 2*k*k offsets + k*k mask per position
            self.offset_conv = nn.Conv2d(
                in_ch,
                3 * kernel_size * kernel_size,  # 2 for offset xy + 1 for mask
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            )
            nn.init.zeros_(self.offset_conv.weight)
            nn.init.zeros_(self.offset_conv.bias)
        else:
            # Fallback: plain conv
            self.fallback = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not HAS_DCN:
            return self.fallback(x)

        out = self.offset_conv(x)
        n = self.k * self.k
        offset = out[:, :2 * n, :, :]                         # (B, 2k², H, W)
        mask = torch.sigmoid(out[:, 2 * n:, :, :])            # (B, k², H, W)

        return deform_conv2d(
            x, offset, self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask,
        )


class DeformBottleneck(nn.Module):
    """
    Bottleneck with deformable 3×3 conv.
    Same interface as ultralytics Bottleneck(c1, c2, shortcut, g, k, e).
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(),
        )
        # Deformable 3×3
        self.dcn = DeformConv2d(c_, c_, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c_)
        self.act2 = nn.SiLU()

        self.cv3 = nn.Sequential(
            nn.Conv2d(c_, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y = self.act2(self.bn2(self.dcn(y)))
        y = self.cv3(y)
        return x + y if self.add else y


class DeformC2f(nn.Module):
    """
    Drop-in replacement for ultralytics C2f.
    Uses DeformBottleneck instead of standard Bottleneck.

    Interface matches C2f(c1, c2, n, shortcut, g, e) exactly.

    Usage in yaml (neck only — see holoYOLO.yaml):
        - [[-1, 6], 1, DeformC2f, [256, False]]
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)

        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.m = nn.ModuleList(
            DeformBottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))
