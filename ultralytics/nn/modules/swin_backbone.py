"""
ultralytics/nn/modules/swin_backbone.py
========================================
NEW file — add to your fork alongside your existing swin_transformer.py.

Adds four thin classes built entirely on top of your existing:
  - SwinTransformerBlock   (your class, untouched)
  - Conv                   (ultralytics standard)

Nothing from timm or torchvision is imported.

Classes exported:
  SwinPatchEmbed   – stride-4 patch partition (Conv2d + LayerNorm)
  SwinPatchMerge   – 2x downsampler between stages (standard PatchMerging)
  SwinStage        – one stage = your SwinTransformerBlock, YAML-friendly
  SwinBackbone     – full 4-stage backbone, returns list[P2,P3,P4,P5]
  SwinSelect       – picks one tensor from that list, exposes c2 for parse_model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Your existing block — we call it, never rewrite it
from .swin_transformer import SwinTransformerBlock
from .conv import Conv

__all__ = [
    "SwinPatchEmbed",
    "SwinPatchMerge",
    "SwinStage",
    "SwinBackbone",
    "SwinSelect",
]


# ─────────────────────────────────────────────────────────────────────────────
class SwinPatchEmbed(nn.Module):
    """
    Patch partition + linear projection — Stage 0 of the Swin hierarchy.

    Splits the image into non-overlapping 4×4 patches and projects each
    patch to embed_dim channels.  Implemented as a single strided Conv2d
    (mathematically identical to the paper's linear layer on flattened patches)
    followed by LayerNorm in channel-last, then back to BCHW.

    Input  : (B,  3,       H,    W)
    Output : (B, embed,   H/4,  W/4)
    """

    def __init__(self, c1: int = 3, embed_dim: int = 96):
        super().__init__()
        self.proj = nn.Conv2d(c1, embed_dim, kernel_size=4, stride=4, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                           # B, C, H/4, W/4
        B, C, H, W = x.shape
        # LayerNorm over channel dim (last dim in token-format)
        x = x.flatten(2).transpose(1, 2)           # B, H*W, C
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)     # B, C, H/4, W/4
        return x


# ─────────────────────────────────────────────────────────────────────────────
class SwinPatchMerge(nn.Module):
    """
    Standard Swin PatchMerging — 2× spatial downsampling between stages.

    Concatenates the 4 spatially interleaved sub-grids of the input along
    the channel dimension (→ 4C), normalises, then linearly projects to 2C.
    No Conv is used here because the spatial neighbourhood is handled by the
    sub-grid pick, not a sliding kernel.

    Input  : (B,  C,  H,   W)
    Output : (B, 2C,  H/2, W/2)
    """

    def __init__(self, c1: int):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * c1)
        self.reduction = nn.Linear(4 * c1, 2 * c1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # pad to even dimensions if needed
        if H % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))   # pad bottom
        if W % 2 == 1:
            x = F.pad(x, (0, 1, 0, 0))   # pad right

        x = x.permute(0, 2, 3, 1)        # B, H, W, C  (channel-last)
        # pick 4 interleaved sub-grids
        x0 = x[:, 0::2, 0::2, :]         # top-left
        x1 = x[:, 1::2, 0::2, :]         # bottom-left
        x2 = x[:, 0::2, 1::2, :]         # top-right
        x3 = x[:, 1::2, 1::2, :]         # bottom-right
        x  = torch.cat([x0, x1, x2, x3], dim=-1)  # B, H/2, W/2, 4C
        x  = self.norm(x)
        x  = self.reduction(x)            # B, H/2, W/2, 2C
        x  = x.permute(0, 3, 1, 2).contiguous()  # B, 2C, H/2, W/2
        return x


# ─────────────────────────────────────────────────────────────────────────────
class SwinStage(nn.Module):
    """
    One Swin stage: wraps YOUR SwinTransformerBlock so parse_model can
    instantiate it from plain YAML args.

    YAML args  : [c2, num_layers, window_size]
    parse_model: passes (c1, c2, num_layers, window_size)

    num_heads is auto-computed as  max(1, c2 // 32)
    — same formula as your SwinTransformer / SwinTransformerB / SwinTransformerC.

    Input / Output : (B, c2, H, W)
    If c1 != c2 your SwinTransformerBlock inserts a Conv internally.
    """

    def __init__(self, c1: int, c2: int, num_layers: int = 2,
                 window_size: int = 7):
        super().__init__()
        num_heads = max(1, c2 // 32)
        # Directly call YOUR block — zero reimplementation
        self.block = SwinTransformerBlock(
            c1=c1, c2=c2,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
class SwinBackbone(nn.Module):
    """
    Complete 4-stage Swin backbone built from your SwinTransformerBlock.

    Used as a SINGLE layer in YAML (backbone layer 0).
    Returns a Python list of 4 BCHW tensors so each scale can be routed
    independently in the head.

    Architecture
    ─────────────────────────────────────────────────────────────────────
    PatchEmbed  (3 → C,  stride 4)
    SwinStage-1  (C  → C,  d[0] blocks)   → P2  (stride  4)
    PatchMerge  (C  → 2C, stride 8)
    SwinStage-2  (2C → 2C, d[1] blocks)   → P3  (stride  8)
    PatchMerge  (2C → 4C, stride 16)
    SwinStage-3  (4C → 4C, d[2] blocks)   → P4  (stride 16)
    PatchMerge  (4C → 8C, stride 32)
    SwinStage-4  (8C → 8C, d[3] blocks)   → P5  (stride 32)
    ─────────────────────────────────────────────────────────────────────

    YAML layer:
        - [-1, 1, SwinBackbone, [embed_dim, depths, window_size]]
        e.g.
        - [-1, 1, SwinBackbone, [96,  [2, 2,  6, 2], 7]]   # Swin-T
        - [-1, 1, SwinBackbone, [96,  [2, 2, 18, 2], 7]]   # Swin-S
        - [-1, 1, SwinBackbone, [128, [2, 2, 18, 2], 7]]   # Swin-B
        - [-1, 1, SwinBackbone, [192, [2, 2, 18, 2], 7]]   # Swin-L

    parse_model passes: (c1=3, embed_dim, depths, window_size)

    Attribute
        self.out_channels : [C, 2C, 4C, 8C]  — queried by parse_model
                            to set c2 = out_channels[-1] for this layer.
    """

    def __init__(
        self,
        c1: int,                        # passed by parse_model (= 3, image ch)
        embed_dim: int        = 96,
        depths: list          = None,
        window_size: int      = 7,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 6, 2]       # Swin-T default

        C = embed_dim
        ws = window_size
        self.out_channels = [C, 2 * C, 4 * C, 8 * C]

        # ── Stage 0: patch embedding ──────────────────────────────────────
        self.patch_embed = SwinPatchEmbed(c1=c1, embed_dim=C)

        # ── Stage 1 → P2  (stride 4) ─────────────────────────────────────
        self.stage1 = SwinStage(c1=C,     c2=C,     num_layers=depths[0], window_size=ws)

        # ── Merge 1 + Stage 2 → P3  (stride 8) ──────────────────────────
        self.merge1 = SwinPatchMerge(c1=C)
        self.stage2 = SwinStage(c1=2*C,   c2=2*C,   num_layers=depths[1], window_size=ws)

        # ── Merge 2 + Stage 3 → P4  (stride 16) ─────────────────────────
        self.merge2 = SwinPatchMerge(c1=2*C)
        self.stage3 = SwinStage(c1=4*C,   c2=4*C,   num_layers=depths[2], window_size=ws)

        # ── Merge 3 + Stage 4 → P5  (stride 32) ─────────────────────────
        self.merge3 = SwinPatchMerge(c1=4*C)
        self.stage4 = SwinStage(c1=8*C,   c2=8*C,   num_layers=depths[3], window_size=ws)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:   x  (B, 3, H, W)
        Returns list of 4 BCHW tensors:
                [P2(C), P3(2C), P4(4C), P5(8C)]
        """
        x  = self.patch_embed(x)    # B, C,   H/4,  W/4

        p2 = self.stage1(x)         # B, C,   H/4,  W/4

        x  = self.merge1(p2)        # B, 2C,  H/8,  W/8
        p3 = self.stage2(x)         # B, 2C,  H/8,  W/8

        x  = self.merge2(p3)        # B, 4C,  H/16, W/16
        p4 = self.stage3(x)         # B, 4C,  H/16, W/16

        x  = self.merge3(p4)        # B, 8C,  H/32, W/32
        p5 = self.stage4(x)         # B, 8C,  H/32, W/32

        return [p2, p3, p4, p5]


# ─────────────────────────────────────────────────────────────────────────────
class SwinSelect(nn.Module):
    """
    Selects one scale tensor from the list produced by SwinBackbone.

    Why not use ultralytics' built-in Index?
    Because Index does not communicate output channel count back to
    parse_model's channel-tracking dict `ch`.  SwinSelect declares
    c2 = args[0] so the downstream Conv knows its c1 automatically.

    YAML args : [out_channels, stage_index]

        - [0, 1, SwinSelect, [96,  0]]   # P2  96ch   (swin-t)
        - [0, 1, SwinSelect, [192, 1]]   # P3 192ch
        - [0, 1, SwinSelect, [384, 2]]   # P4 384ch
        - [0, 1, SwinSelect, [768, 3]]   # P5 768ch

    parse_model passes: (c1, c2, stage)
        c1    = ch[0] = 768  (last stage of backbone — irrelevant here)
        c2    = args[0]      = channel count of the selected stage
        stage = args[1]      = index into the list (0-3)
    """

    def __init__(self, c1: int, c2: int, stage: int):
        super().__init__()
        self.stage       = stage
        self.out_channels = c2   # parse_model reads this to set c2 for this layer

    def forward(self, feature_list):
        return feature_list[self.stage]
