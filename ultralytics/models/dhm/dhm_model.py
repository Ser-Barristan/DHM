"""
models/dhm_model.py
===================
DHMPhaseNet — full model wrapping Ultralytics backbone/neck/head
with RS-FiLM side-channel conditioning.

INSTRUCTION: Place this file at
    ultralytics/models/dhm/model.py
and create ultralytics/models/dhm/__init__.py importing DHMPhaseNet.

Or simply import it from the Jupyter notebook — it does NOT need to live
inside the Ultralytics package tree to work.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import (
    PatchEmbedDHM,
    PatchMergingDHM,
    MWSwinStageDHM,
    BiFPNLayerDHM,
    RSFiLMGenerator,
)
from ultralytics.nn.modules.head import PhaseRegressionHead


class DHMPhaseNet(nn.Module):
    """
    Multi-Window SwinTransformer backbone
    → BiFPN neck
    → ASPP + Laplacian Pyramid phase regression head
    + RS-FiLM physics conditioning (Rayleigh-Sommerfeld features)

    Architecture summary
    --------------------
    Input      768×768 × 1ch  hologram
    PatchEmbed  patch=4  →  192×192 × 96ch tokens
    Stage 0    ws=4,  heads=3,  depth=2  →  192×192 × 96   + merge → 96×96 × 192
    Stage 1    ws=8,  heads=6,  depth=2  →  96×96  × 192   + merge → 48×48 × 384
    Stage 2    ws=12, heads=12, depth=6  →  48×48  × 384   (no merge)
    BiFPN      [P3(192²,96), P4(96²,192), P5(48²,384)] → all (256ch)
    Head       ASPP × 3 + Laplacian pyramid → 768×768 × 1ch phase

    FiLM conditioning
    -----------------
    RSFiLMGenerator maps (λ, L, z) → per-stage (γ, β) via 8 RS features:
        obliquity z/r_max, log(φ_corner), log(φ_onaxis),
        log(z/λ), log(M), log(px_obj), sin/cos(φ_onaxis mod 2π)
    Each Swin stage applies FiLM after its final block.

    Args
    ----
    embed_dim      : int   patch embedding dimension (default 96)
    window_sizes   : tuple window sizes per stage     (default (4, 8, 12))
    num_heads      : tuple attention heads per stage  (default (3, 6, 12))
    depths         : tuple block depth per stage      (default (2, 2, 6))
    neck_dim       : int   BiFPN unified channels      (default 256)
    aspp_dim       : int   ASPP output channels        (default 128)
    mid_channels   : int   Lap-decoder intermediate    (default 128)
    film_hidden    : int   FiLM MLP hidden dim         (default 128)
    pixel_size     : float detector pixel pitch μm     (default 3.8e-3 = 3.8 μm in mm)
                           NOTE: kept in mm internally to match L_value/z_value units
    n_pix          : int   image side length            (default 768)
    drop_rate      : float dropout in Swin blocks       (default 0.)
    attn_drop_rate : float attention dropout            (default 0.)
    drop_path_rate : float stochastic depth max rate    (default 0.1)
    """

    def __init__(
        self,
        embed_dim: int        = 96,
        window_sizes: tuple   = (4, 8, 12),
        num_heads: tuple      = (3, 6, 12),
        depths: tuple         = (2, 2, 6),
        neck_dim: int         = 256,
        aspp_dim: int         = 128,
        mid_channels: int     = 128,
        film_hidden: int      = 128,
        pixel_size: float     = 3.8e-3,   # mm  (3.8 μm)
        n_pix: int            = 768,
        drop_rate: float      = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        assert len(window_sizes) == len(num_heads) == len(depths) == 3, \
            "Exactly 3 stages expected"

        self.pixel_size = pixel_size   # mm
        self.n_pix      = n_pix

        stage_dims = [embed_dim, embed_dim * 2, embed_dim * 4]  # 96,192,384

        # ── patch embedding ──────────────────────────────────────────────
        self.patch_embed = PatchEmbedDHM(
            in_chans=1, embed_dim=embed_dim, patch_size=4
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ── stochastic depth rates ───────────────────────────────────────
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        d0, d1, d2 = depths

        # ── backbone stages ──────────────────────────────────────────────
        self.stage0 = MWSwinStageDHM(
            dim=stage_dims[0], num_heads=num_heads[0],
            window_size=window_sizes[0], depth=depths[0],
            downsample=True, mlp_ratio=4.,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[:d0],
        )
        self.stage1 = MWSwinStageDHM(
            dim=stage_dims[1], num_heads=num_heads[1],
            window_size=window_sizes[1], depth=depths[1],
            downsample=True, mlp_ratio=4.,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[d0:d0+d1],
        )
        self.stage2 = MWSwinStageDHM(
            dim=stage_dims[2], num_heads=num_heads[2],
            window_size=window_sizes[2], depth=depths[2],
            downsample=False, mlp_ratio=4.,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[d0+d1:],
        )

        # ── RS-FiLM generator ────────────────────────────────────────────
        # pixel_size passed to RS features as μm (×1000 from mm)
        self.film = RSFiLMGenerator(
            physics_dim=8,
            hidden_dim=film_hidden,
            stage_dims=stage_dims,      # [96,192,384]
            pixel_size=pixel_size * 1e3,  # convert to μm for RS formula
            n_pix=n_pix,
        )

        # ── BiFPN neck ───────────────────────────────────────────────────
        self.bifpn = BiFPNLayerDHM(
            in_dims=tuple(stage_dims), neck_dim=neck_dim
        )

        # ── regression head ──────────────────────────────────────────────
        self.head = PhaseRegressionHead(
            nc=1,
            neck_dim=neck_dim,
            aspp_dim=aspp_dim,
            mid_channels=mid_channels,
        )

        self._init_weights()

    # ── weight init ──────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, img, wavelength, L_value, z_value):
        """
        Parameters
        ----------
        img        : (B, 1, 768, 768)  normalised hologram
        wavelength : (B,)  μm
        L_value    : (B,)  mm
        z_value    : (B,)  mm

        Returns
        -------
        phase : (B, 1, 768, 768)  predicted phase map (unbounded float)
        """
        # ── RS-FiLM features ─────────────────────────────────────────────
        # Returns list of 3 (gamma, beta) pairs, one per stage
        film_params = self.film(wavelength, L_value, z_value)

        # ── patch embedding ──────────────────────────────────────────────
        x, H, W = self.patch_embed(img)   # (B, 192*192, 96), H=192, W=192
        x = self.pos_drop(x)

        # ── stage 0  ws=4 ────────────────────────────────────────────────
        g0, b0 = film_params[0]
        x0, H0, W0, x_d0, H1, W1 = self.stage0(x,    H,  W,  g0, b0)
        # x0:   (B, 192*192, 96)  — pre-merge feature
        # x_d0: (B,  96*96, 192)  — post-merge for next stage

        # ── stage 1  ws=8 ────────────────────────────────────────────────
        g1, b1 = film_params[1]
        x1, H1, W1, x_d1, H2, W2 = self.stage1(x_d0, H1, W1, g1, b1)

        # ── stage 2  ws=12 (no downsample) ───────────────────────────────
        g2, b2 = film_params[2]
        x2, H2, W2, _,    _,  _  = self.stage2(x_d1, H2, W2, g2, b2)

        # ── reshape token sequences → spatial feature maps ───────────────
        B = img.shape[0]
        P3 = x0.transpose(1,2).view(B,  96, H0, W0)  # (B, 96, 192, 192)
        P4 = x1.transpose(1,2).view(B, 192, H1, W1)  # (B,192,  96,  96)
        P5 = x2.transpose(1,2).view(B, 384, H2, W2)  # (B,384,  48,  48)

        # ── BiFPN neck ───────────────────────────────────────────────────
        neck_feats = self.bifpn([P3, P4, P5])         # list of 3 × (B,256,H,W)

        # ── phase regression head ─────────────────────────────────────────
        phase = self.head(neck_feats)                 # (B, 1, 768, 768)
        return phase

    # ── parameter count helper ───────────────────────────────────────────────
    def info(self):
        n  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DHMPhaseNet  trainable parameters: {n/1e6:.2f} M")
        stages = {
            'PatchEmbed': self.patch_embed,
            'Stage0 (ws=4)': self.stage0,
            'Stage1 (ws=8)': self.stage1,
            'Stage2 (ws=12)': self.stage2,
            'RSFiLM': self.film,
            'BiFPN': self.bifpn,
            'Head (ASPP+Lap)': self.head,
        }
        for name, mod in stages.items():
            n_mod = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            print(f"  {name:<22s}  {n_mod/1e6:6.2f} M")


# ── quick sanity check ───────────────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = DHMPhaseNet().to(device)
    model.info()

    B  = 2
    img = torch.randn(B, 1, 768, 768).to(device)
    wl  = torch.tensor([0.405, 0.405]).to(device)
    L   = torch.tensor([18.96, 18.96]).to(device)
    z   = torch.tensor([0.36,  0.72]).to(device)

    with torch.no_grad():
        out = model(img, wl, L, z)
    print(f"Output shape : {out.shape}")   # (2, 1, 768, 768)
    print(f"Output range : [{out.min():.3f}, {out.max():.3f}]")
