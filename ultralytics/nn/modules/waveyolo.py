"""
WaveYOLO custom modules.
File location: ultralytics/nn/modules/waveyolo.py  (NEW FILE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .conv import Conv  # ultralytics built-in


# ──────────────────────────────────────────────────────────────────────────────
# 1. Annular depthwise conv  (ring-shaped kernel mask)
# ──────────────────────────────────────────────────────────────────────────────

class AnnularDWConv(nn.Module):
    """Depthwise conv with a fixed annular kernel mask.

    Forces filters to respond to ring/halo edge patterns rather than
    solid blobs.  The mask is re-parameterisable at inference:
        w_eff = weight * mask   (done in forward; fuse for export)

    Args:
        c        : number of channels
        k        : kernel size (odd; default 7)
        r_inner  : inner radius of the ring mask (default 2)
        r_outer  : outer radius of the ring mask (default 3)
    """

    def __init__(self, c: int, k: int = 7, r_inner: int = 2, r_outer: int = 3):
        super().__init__()
        assert k % 2 == 1, "kernel size must be odd"
        self.padding = k // 2
        self.groups  = c
        self.conv    = nn.Conv2d(c, c, k, padding=k // 2,
                                 groups=c, bias=False)
        mask = self._make_ring_mask(k, r_inner, r_outer)
        self.register_buffer('mask', mask.view(1, 1, k, k))

    @staticmethod
    def _make_ring_mask(k: int, r_inner: float, r_outer: float) -> torch.Tensor:
        cy = cx = k // 2
        y, x = torch.meshgrid(torch.arange(k), torch.arange(k), indexing='ij')
        dist = ((x - cx).float() ** 2 + (y - cy).float() ** 2).sqrt()
        return ((dist >= r_inner) & (dist <= r_outer)).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv.weight * self.mask   # masked weights
        return F.conv2d(x, w, stride=1,
                        padding=self.padding, groups=x.shape[1])


# ──────────────────────────────────────────────────────────────────────────────
# 2. C2f-Ring  (drop-in for C2f; adds annular branch)
# ──────────────────────────────────────────────────────────────────────────────

class C2f_Ring(nn.Module):
    """C2f with a parallel AnnularDWConv branch injected into the last split.

    Inherits the CSP-bottleneck gradient flow of standard C2f while adding
    ring-edge inductive bias via the annular branch.

    Usage in YAML:  replace  C2f  with  C2f_Ring  (same signature).
    """

    def __init__(self, c1: int, c2: int, n: int = 1,
                 shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c   = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m   = nn.ModuleList(
            Conv(self.c, self.c, 3) for _ in range(n)
        )
        # ── novel ring branch ──────────────────────────────────────────
        self.ring      = AnnularDWConv(self.c, k=7, r_inner=2, r_outer=3)
        self.ring_proj = Conv(self.c, self.c, 1)
        self.ring_bn   = nn.BatchNorm2d(self.c)
        # ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        # Inject ring features additively into the last branch tensor
        ring_feat = self.ring_proj(
            F.silu(self.ring_bn(self.ring(y[-1])))
        )
        y[-1] = y[-1] + ring_feat
        return self.cv2(torch.cat(y, 1))


# ──────────────────────────────────────────────────────────────────────────────
# 3. 2-D Haar wavelet  (differentiable, no learned params)
# ──────────────────────────────────────────────────────────────────────────────

class HaarWavelet2D(nn.Module):
    """Fixed-weight 2-D Haar DWT.

    Returns four sub-bands  (LL, LH, HL, HH)  each at half spatial
    resolution.  HH captures diagonal ring edges; used by WaveFPN.
    """

    def __init__(self):
        super().__init__()
        h = torch.tensor([[1., 1.], [1., 1.]]) / 2.
        g = torch.tensor([[1., -1.], [1., -1.]]) / 2.
        filters = torch.stack([
            torch.kron(h, h),   # LL
            torch.kron(h, g),   # LH
            torch.kron(g, h),   # HL
            torch.kron(g, g),   # HH  ← ring edges
        ]).unsqueeze(1)          # (4, 1, 2, 2)
        self.register_buffer('filters', filters)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)
        out    = F.conv2d(x_flat, self.filters, stride=2)   # (B*C, 4, H/2, W/2)
        out    = out.reshape(B, C, 4, H // 2, W // 2)
        return out.unbind(2)   # → LL, LH, HL, HH


# ──────────────────────────────────────────────────────────────────────────────
# 4. WaveFPN  (wavelet-guided feature fusion, replaces simple upsample+cat)
# ──────────────────────────────────────────────────────────────────────────────

class WaveFPN(nn.Module):
    """Fuse a coarse feature map with a fine one using HH wavelet guidance.

    Instead of bilinear upsampling (which blurs ring edges), the coarse
    map is upsampled and then **gated** by the HH sub-band of the fine map.
    This routes HF ring boundary energy from the fine scale into the fused
    feature without requiring extra parameters beyond two 1×1 convs.

    Args:
        c : number of channels in BOTH feat_coarse and feat_fine
            (they must match — add a Conv before this layer if needed)
    """

    def __init__(self, c: int):
        super().__init__()
        self.haar   = HaarWavelet2D()
        # Project [upsampled_coarse ‖ HH_fine] → c
        self.fuse   = Conv(c * 2, c, 1)
        self.gate   = nn.Sequential(
            Conv(c, c // 4, 1),
            nn.SiLU(),
            Conv(c // 4, c, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat_coarse: torch.Tensor,
                feat_fine: torch.Tensor) -> torch.Tensor:
        """
        feat_coarse : (B, C, H/2, W/2)
        feat_fine   : (B, C, H,   W  )
        returns     : (B, C, H,   W  )
        """
        _, _, _, HH = self.haar(feat_fine)
        up   = F.interpolate(feat_coarse, scale_factor=2,
                             mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([up, HH], dim=1))
        gate  = self.gate(HH)
        return fused * gate + fused   # residual-gated


# ──────────────────────────────────────────────────────────────────────────────
# 5. PSA-Radial  (polar self-attention; replaces PSA in backbone bottleneck)
# ──────────────────────────────────────────────────────────────────────────────

class PSARadial(nn.Module):
    """Polar-position-aware multi-head self-attention.

    Encodes each spatial position as (r, θ) instead of Cartesian (x, y).
    This gives the attention map an explicit radial symmetry inductive bias,
    helping it attend to ring structures regardless of orientation.

    Args:
        c         : channel dim (must equal backbone output channels)
        num_heads : MHA heads (default 8)
        drop      : attention dropout (default 0.0)
    """

    def __init__(self, c: int, num_heads: int = 8, drop: float = 0.0):
        super().__init__()
        self.c         = c
        self.attn      = nn.MultiheadAttention(c, num_heads,
                                               dropout=drop,
                                               batch_first=True)
        self.norm1     = nn.LayerNorm(c)
        self.norm2     = nn.LayerNorm(c)
        self.ffn       = nn.Sequential(
            nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c)
        )
        self._pe_cache: dict = {}

    def _polar_pe(self, H: int, W: int,
                  device: torch.device) -> torch.Tensor:
        key = (H, W, str(device))
        if key not in self._pe_cache:
            cy, cx = H / 2., W / 2.
            y  = torch.arange(H, device=device, dtype=torch.float32)
            x  = torch.arange(W, device=device, dtype=torch.float32)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            r     = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt() / (max(H, W) / 2.)
            theta = torch.atan2(yy - cy, xx - cx)     # [-π, π]

            d    = self.c // 4
            freq = torch.arange(d, device=device, dtype=torch.float32)
            base = 1000. ** (2. * freq / d)            # (d,)

            def sincos(v):   # v: (H,W) → (H*W, 2d)
                v_flat = v.flatten().unsqueeze(-1)     # (HW, 1)
                return torch.cat([
                    torch.sin(v_flat / base.unsqueeze(0)),
                    torch.cos(v_flat / base.unsqueeze(0)),
                ], dim=-1)

            pe = sincos(r) + sincos(theta)             # (HW, 2d) ← 2d == c//2
            # Pad to full c if needed
            if pe.shape[-1] < self.c:
                pad = torch.zeros(H * W, self.c - pe.shape[-1], device=device)
                pe  = torch.cat([pe, pad], dim=-1)
            self._pe_cache[key] = pe[:, :self.c]       # (HW, c)
        return self._pe_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        flat = x.flatten(2).permute(0, 2, 1)           # (B, HW, C)
        pe   = self._polar_pe(H, W, x.device).unsqueeze(0)   # (1, HW, C)
        qk   = flat + pe                                # add polar encoding to Q,K only
        # Self-attention with residual
        attn_out, _ = self.attn(qk, qk, flat)
        flat = self.norm1(flat + attn_out)
        flat = self.norm2(flat + self.ffn(flat))
        return flat.permute(0, 2, 1).reshape(B, C, H, W)
