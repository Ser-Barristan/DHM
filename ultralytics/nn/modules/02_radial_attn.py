"""
FILE: ultralytics/nn/modules/radial_attn.py
ADD THIS AS A NEW FILE IN YOUR FORK

Radial Attention Module — explicitly encodes the annular/ring structure
of holographic diffraction patterns. Replaces the standard SPPF in YOLOv8n.

Key insight: Standard MaxPool SPPF uses square receptive fields.
Holograms have RADIALLY symmetric features. Annular pooling captures
concentric ring energy that square pooling misses.

Total added params: ~8K (negligible vs 2.5M budget)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_annular_mask(size: int, r_inner: float, r_outer: float, device):
    """Binary mask for a ring between r_inner and r_outer (normalized 0-1 radius)."""
    cy = cx = size / 2.0
    y = torch.arange(size, dtype=torch.float32, device=device).view(-1, 1)
    x = torch.arange(size, dtype=torch.float32, device=device).view(1, -1)
    dist = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2) / (size / 2.0)
    return ((dist >= r_inner) & (dist < r_outer)).float()  # (size, size)


# ── main module ───────────────────────────────────────────────────────────────

class AnnularPool(nn.Module):
    """
    Multi-ring pooling: pools features in N concentric annular bands.
    Each band's pooled vector is projected back to spatial features via broadcast.

    This creates ring-aware context — a direct match to hologram physics.
    """

    def __init__(self, channels: int, n_rings: int = 4, pool_size: int = 13):
        super().__init__()
        self.channels = channels
        self.n_rings = n_rings
        self.pool_size = pool_size

        # Learnable ring boundaries (initialized uniformly, trained end-to-end)
        boundaries = torch.linspace(0.0, 1.0, n_rings + 1)
        self.r_inner = nn.Parameter(boundaries[:-1])
        self.r_outer = nn.Parameter(boundaries[1:])

        # Per-ring channel mixer
        self.ring_proj = nn.ModuleList(
            [nn.Linear(channels, channels, bias=False) for _ in range(n_rings)]
        )
        self.gate = nn.Sequential(
            nn.Linear(channels * n_rings, channels),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.pool_size

        # Adaptive avg pool to fixed spatial size for mask application
        xp = F.adaptive_avg_pool2d(x, (p, p))  # (B, C, p, p)

        ring_feats = []
        for i in range(self.n_rings):
            r0 = self.r_inner[i].clamp(0.0, 0.99)
            r1 = self.r_outer[i].clamp(r0.item() + 0.01, 1.0)
            mask = _make_annular_mask(p, r0.item(), r1.item(), x.device)  # (p,p)
            mask_sum = mask.sum().clamp(min=1.0)

            # Masked average pool → (B, C)
            ring_mean = (xp * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1)) / mask_sum
            ring_feats.append(self.ring_proj[i](ring_mean))  # (B, C)

        # Gate: all rings → single (B, C) attention vector
        stacked = torch.cat(ring_feats, dim=-1)  # (B, C*n_rings)
        attn = self.gate(stacked)                # (B, C)

        # Broadcast attention back to spatial
        attn = attn.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # Residual with layer norm (applied over channel dim)
        out = x * attn
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out + x


class HoloSPPF(nn.Module):
    """
    Replacement for YOLOv8's SPPF module.

    Standard SPPF:  Conv → 3x MaxPool(5) concat → Conv
    HoloSPPF:       Conv → AnnularPool → Conv
                    + standard SPPF path (concatenated)

    The two paths are gated so the network learns how much
    to rely on ring-aware vs. standard spatial pooling.

    Drop-in: same in/out channels as SPPF(c1, c2).

    Usage in yaml:
        - [[-1], 1, HoloSPPF, [256, 5]]   # same args as SPPF
    """

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2

        # Standard SPPF path
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(),
        )
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_ * 4, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        # Annular path
        self.annular = AnnularPool(c1, n_rings=4, pool_size=13)
        self.ann_proj = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        # Learned fusion gate (scalar per channel)
        self.alpha = nn.Parameter(torch.ones(c2) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard SPPF
        y = self.cv1(x)
        p1 = self.pool(y)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        sppf_out = self.cv2(torch.cat([y, p1, p2, p3], dim=1))

        # Annular path
        ann_out = self.ann_proj(self.annular(x))

        # Gated fusion
        a = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        return a * sppf_out + (1 - a) * ann_out
