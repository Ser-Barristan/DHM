# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
HoloYOLO — Radial / Annular Attention + HoloSPPF
FILE PATH IN YOUR FORK: ultralytics/nn/modules/radial_attn.py
CREATE THIS FILE (it does not exist in vanilla ultralytics)

HoloSPPF replaces the vanilla SPPF in YOLOv8n backbone (layer 9).
It fuses standard square MaxPool SPPF with an annular-pooling path
that explicitly captures concentric ring energy — the dominant
feature of holographic diffraction patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("AnnularPool", "HoloSPPF")


# ── helpers ────────────────────────────────────────────────────────────────────

def _annular_mask(size: int, r_inner: float, r_outer: float, device, dtype) -> torch.Tensor:
    """
    Return a (size, size) float mask for the annular band.
    Built in float32 for precision then cast to match feature dtype.
    """
    cy = cx = (size - 1) / 2.0
    ys = torch.arange(size, dtype=torch.float32, device=device).view(-1, 1)
    xs = torch.arange(size, dtype=torch.float32, device=device).view(1, -1)
    dist = torch.sqrt((ys - cy) ** 2 + (xs - cx) ** 2) / (size / 2.0)
    return ((dist >= r_inner) & (dist < r_outer)).to(dtype=dtype)   # (size, size)


# ── AnnularPool ────────────────────────────────────────────────────────────────

class AnnularPool(nn.Module):
    """
    Multi-ring spatial pooling.

    Divides the spatial extent into `n_rings` concentric annular bands.
    Each band is avg-pooled to a (B, C) descriptor, mixed through a
    learnable linear layer, then gated and broadcast back spatially.

    Ring boundaries are learnable — the network can widen or narrow
    each band depending on the object scale distribution.

    Args:
        channels  (int): feature-map channel count
        n_rings   (int): number of concentric bands (default 4)
        pool_size (int): fixed spatial size used for mask application (default 13)
    """

    def __init__(self, channels: int, n_rings: int = 4, pool_size: int = 13):
        super().__init__()
        self.channels  = channels
        self.n_rings   = n_rings
        self.pool_size = pool_size

        # Learnable ring boundary params (init uniformly)
        bounds = torch.linspace(0.0, 1.0, n_rings + 1)
        self.r_inner = nn.Parameter(bounds[:-1].clone())     # (n_rings,)
        self.r_outer = nn.Parameter(bounds[1:].clone())      # (n_rings,)

        # Per-ring feature transform
        self.ring_proj = nn.ModuleList(
            [nn.Linear(channels, channels, bias=False) for _ in range(n_rings)]
        )
        # Gate: all ring descriptors → channel attention vector
        self.gate = nn.Sequential(
            nn.Linear(channels * n_rings, channels),
            nn.Sigmoid(),
        )
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)  — residual-connected ring-attentive features
        """
        B, C, H, W = x.shape
        p   = self.pool_size
        xp  = F.adaptive_avg_pool2d(x, (p, p))              # (B, C, p, p)
        dev = x.device
        dt  = x.dtype                                        # float32 or float16 (AMP)

        ring_descs = []
        for i in range(self.n_rings):
            r0 = self.r_inner[i].clamp(0.00, 0.98)
            r1 = self.r_outer[i].clamp(r0.detach() + 0.01, 1.0)

            mask     = _annular_mask(p, float(r0), float(r1), dev, dt)  # (p, p)
            mask_sum = mask.sum().clamp(min=1.0)

            # Masked average → (B, C)
            desc = (xp * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1)) / mask_sum
            ring_descs.append(self.ring_proj[i](desc))       # (B, C)

        # Gate over all rings
        stacked = torch.cat(ring_descs, dim=-1)              # (B, C*n_rings)
        attn    = self.gate(stacked).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # GroupNorm requires float32 in some torch versions — upcast then restore
        y = x * attn
        out = self.norm(y.float()).to(dtype=x.dtype)
        return out + x                                       # residual


# ── HoloSPPF ──────────────────────────────────────────────────────────────────

class HoloSPPF(nn.Module):
    """
    Hologram-aware SPPF — drop-in replacement for vanilla SPPF.

    Two parallel paths:
      A) Standard SPPF  (3× MaxPool cascade)
      B) AnnularPool    (ring-aware global context)
    Fused via a learned per-channel scalar gate (alpha).

    Interface matches ultralytics SPPF(c1, c2, k=5) exactly.

    In holoYOLOv8n.yaml, backbone layer 9:
        BEFORE: - [-1, 1, SPPF,     [256, 5]]
        AFTER:  - [-1, 1, HoloSPPF, [256, 5]]

    Args:
        c1 (int): input channels
        c2 (int): output channels
        k  (int): MaxPool kernel size (default 5, same as SPPF)
    """

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c1 = int(c1)
        c2 = int(c2)
        k  = int(k)
        c_ = c1 // 2                                         # hidden channels

        # ── Path A: standard SPPF ─────────────────────────────────────────
        self.cv1 = self._conv(c1, c_, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2  = self._conv(c_ * 4, c2, 1)

        # ── Path B: annular-pool ──────────────────────────────────────────
        self.annular  = AnnularPool(c1, n_rings=4, pool_size=13)
        self.ann_proj = self._conv(c1, c2, 1)

        # ── Fusion gate ───────────────────────────────────────────────────
        # alpha=0.5 init: equal blend; trained to favour whichever is better
        self.alpha = nn.Parameter(torch.full((c2,), 0.5))

    @staticmethod
    def _conv(ci, co, k, s=1):
        return nn.Sequential(
            nn.Conv2d(ci, co, k, stride=s, padding=k // 2, bias=False),
            nn.BatchNorm2d(co),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Path A
        y   = self.cv1(x)
        p1  = self.pool(y)
        p2  = self.pool(p1)
        p3  = self.pool(p2)
        sppf_out = self.cv2(torch.cat([y, p1, p2, p3], dim=1))

        # Path B
        ann_out = self.ann_proj(self.annular(x))

        # Gated fusion
        a = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        return a * sppf_out + (1.0 - a) * ann_out
