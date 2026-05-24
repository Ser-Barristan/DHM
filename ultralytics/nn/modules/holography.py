# ultralytics/nn/modules/holography.py
# Physics-aware SwinT modules for DLHM phase reconstruction
# Add this file to your fork at ultralytics/nn/modules/holography.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Swin Transformer primitives ───────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches. Supports in_chans=1 (grayscale)."""
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                          # (B, C, H/4, W/4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)          # (B, H*W, C)
        x = self.norm(x)
        return x, H, W


def window_partition(x, window_size):
    """Partition (B, H, W, C) into windows of shape (num_windows*B, ws, ws, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partitioning."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative positional bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, ws, ws)
        coords_flatten = torch.flatten(coords, 1)  # (2, ws*ws)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Single Swin Transformer block with optional shifted window attention."""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()  # simplified; use timm DropPath if available
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, H, W, attn_mask=None):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x, self.window_size)   # (num_win*B, ws, ws, C)
        windows = windows.view(-1, self.window_size ** 2, C)
        attn_windows = self.attn(windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Downsample 2x by merging 2x2 patches."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class SwinStage(nn.Module):
    """One stage of the Swin Transformer (alternating W-MSA and SW-MSA)."""
    def __init__(self, dim, depth, num_heads, window_size=8, downsample=True, drop_path_rate=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                drop_path=drop_path_rate,
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        feat = x.view(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W, feat  # feat is spatial feature before downsampling


class SwinTBackbone(nn.Module):
    """
    Swin Transformer backbone producing 4-scale spatial feature maps.
    Replaces the CSPDarknet backbone in YOLO.
    Input: (B, 1, H, W) grayscale hologram.
    Output: list of (B, C_i, H_i, W_i) at strides [4, 8, 16, 32].

    Args:
        in_chans: 1 for grayscale hologram
        embed_dim: base embedding dimension (96 for small, 128 for base)
        depths: blocks per stage, e.g. [2, 2, 6, 2]
        num_heads: attention heads per stage, e.g. [3, 6, 12, 24]
        window_size: local attention window (8 for 1024x1024 input)
    """
    def __init__(self, in_chans=1, embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=8):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=4, in_chans=in_chans, embed_dim=embed_dim)

        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.out_channels = dims  # [96, 192, 384, 768] for small

        self.stages = nn.ModuleList()
        for i, (d, nh) in enumerate(zip(depths, num_heads)):
            # Last stage: no downsampling (we keep H/32, W/32)
            self.stages.append(
                SwinStage(
                    dim=dims[i],
                    depth=d,
                    num_heads=nh,
                    window_size=window_size,
                    downsample=(i < len(depths) - 1),
                )
            )

    def forward(self, x):
        x, H, W = self.patch_embed(x)   # tokens at stride 4
        feats = []
        for stage in self.stages:
            x, H, W, feat = stage(x, H, W)
            feats.append(feat)           # spatial feature at each stride
        # feats[0]: stride 4, feats[1]: stride 8, feats[2]: stride 16, feats[3]: stride 32
        return feats


# ─── FiLM Conditioning ─────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Maps physics vector → per-channel (gamma, beta) to condition feature maps.
    This is how the model learns physics without explicit equations.
    """
    def __init__(self, physics_dim, num_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(physics_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_channels * 2),
        )

    def forward(self, x, physics):
        # x: (B, C, H, W), physics: (B, physics_dim)
        out = self.net(physics)                              # (B, 2C)
        gamma, beta = out.chunk(2, dim=-1)
        gamma = gamma.view(gamma.shape[0], -1, 1, 1)
        beta  = beta.view(beta.shape[0],  -1, 1, 1)
        return x * (1.0 + gamma) + beta


# ─── PANet Neck (YOLO-style) ───────────────────────────────────────────────────

class ConvBNSiLU(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C2fBlock(nn.Module):
    """YOLOv8 C2f cross-stage partial block."""
    def __init__(self, c1, c2, n=2):
        super().__init__()
        h = c2 // 2
        self.cv1 = ConvBNSiLU(c1, 2 * h, k=1, p=0)
        self.cv2 = ConvBNSiLU((2 + n) * h, c2, k=1, p=0)
        self.m   = nn.ModuleList([
            nn.Sequential(ConvBNSiLU(h, h), ConvBNSiLU(h, h))
            for _ in range(n)
        ])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for blk in self.m:
            y.append(blk(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class PANetNeck(nn.Module):
    """YOLO-style PANet for multi-scale feature fusion."""
    def __init__(self, channels):
        super().__init__()
        c0, c1, c2, c3 = channels

        # Top-down
        self.lat3    = ConvBNSiLU(c3, c2, k=1, p=0)
        self.fuse2td = C2fBlock(c2 + c2, c2)
        self.lat2    = ConvBNSiLU(c2, c1, k=1, p=0)
        self.fuse1td = C2fBlock(c1 + c1, c1)
        self.lat1    = ConvBNSiLU(c1, c0, k=1, p=0)
        self.fuse0td = C2fBlock(c0 + c0, c0)

        # Bottom-up
        self.dn0     = ConvBNSiLU(c0, c0, s=2)
        self.fuse1bu = C2fBlock(c0 + c1, c1)
        self.dn1     = ConvBNSiLU(c1, c1, s=2)
        self.fuse2bu = C2fBlock(c1 + c2, c2)
        self.dn2     = ConvBNSiLU(c2, c2, s=2)
        self.fuse3bu = C2fBlock(c2 + c3, c3)

        self.out_channels = [c0, c1, c2, c3]

    def forward(self, feats):
        f0, f1, f2, f3 = feats
        # top-down
        p2 = self.fuse2td(torch.cat([F.interpolate(self.lat3(f3), size=f2.shape[-2:], mode='nearest'), f2], 1))
        p1 = self.fuse1td(torch.cat([F.interpolate(self.lat2(p2), size=f1.shape[-2:], mode='nearest'), f1], 1))
        p0 = self.fuse0td(torch.cat([F.interpolate(self.lat1(p1), size=f0.shape[-2:], mode='nearest'), f0], 1))
        # bottom-up
        n1 = self.fuse1bu(torch.cat([self.dn0(p0), p1], 1))
        n2 = self.fuse2bu(torch.cat([self.dn1(n1), p2], 1))
        n3 = self.fuse3bu(torch.cat([self.dn2(n2), f3], 1))
        return [p0, n1, n2, n3]


# ─── Phase Reconstruction Head with FiLM ──────────────────────────────────────

class FiLMPhaseDecoder(nn.Module):
    """
    Progressive upsampling decoder with FiLM physics conditioning at each stage.
    Takes PANet outputs and decodes to full-resolution phase map.
    """
    def __init__(self, neck_channels, physics_dim=4):
        super().__init__()
        c0, c1, c2, c3 = neck_channels

        # Decode stride-32 → stride-16 → stride-8 → stride-4
        self.up3   = nn.Sequential(ConvBNSiLU(c3, c2), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.film3 = FiLMLayer(physics_dim, c2)
        self.fuse3 = C2fBlock(c2 + c2, c2)

        self.up2   = nn.Sequential(ConvBNSiLU(c2, c1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.film2 = FiLMLayer(physics_dim, c1)
        self.fuse2 = C2fBlock(c1 + c1, c1)

        self.up1   = nn.Sequential(ConvBNSiLU(c1, c0), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.film1 = FiLMLayer(physics_dim, c0)
        self.fuse1 = C2fBlock(c0 + c0, c0)

        # stride-4 → stride-1 (2x + 2x upsampling)
        self.final = nn.Sequential(
            ConvBNSiLU(c0, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNSiLU(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, neck_feats, physics):
        p0, n1, n2, n3 = neck_feats
        x = self.film3(self.up3(n3), physics)
        x = self.fuse3(torch.cat([x, n2], 1))
        x = self.film2(self.up2(x), physics)
        x = self.fuse2(torch.cat([x, n1], 1))
        x = self.film1(self.up1(x), physics)
        x = self.fuse1(torch.cat([x, p0], 1))
        return self.final(x)   # (B, 1, H, W) in [0,1]
