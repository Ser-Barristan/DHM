# ultralytics/nn/modules/holography.py
"""
Physics-Aware Swin-T + YOLO PANet modules for DLHM holographic phase reconstruction.
No external model library dependency. Pure PyTorch + ultralytics C2f/Conv reuse.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Swin Transformer primitives
# ──────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Stride-4 patch embedding, supports 1-channel (grayscale) input."""
    def __init__(self, in_chans=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                        # (B, C, H/4, W/4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B, H*W, C)
        x = self.norm(x)
        return x, H, W


def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)


def window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowMSA(nn.Module):
    """Window multi-head self-attention with relative position bias."""
    def __init__(self, dim, num_heads, ws, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ws = ws
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # Relative position bias
        self.rel_bias = nn.Parameter(torch.zeros((2*ws-1)**2, num_heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
        coords = torch.stack(torch.meshgrid(
            torch.arange(ws), torch.arange(ws), indexing='ij'))   # (2,ws,ws)
        flat  = coords.flatten(1)                                   # (2,ws*ws)
        rel   = flat[:, :, None] - flat[:, None, :]                 # (2,N,N)
        rel   = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer('rel_idx', rel.sum(-1))               # (N,N)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.rel_bias[self.rel_idx.view(-1)].view(
            self.ws**2, self.ws**2, self.num_heads).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinBlock(nn.Module):
    """One Swin Transformer block (W-MSA or SW-MSA)."""
    def __init__(self, dim, num_heads, ws, shift=False, mlp_ratio=4., drop=0.):
        super().__init__()
        self.ws    = ws
        self.shift = ws // 2 if shift else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowMSA(dim, num_heads, ws, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_h), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_h, dim), nn.Dropout(drop),
        )

    def _make_mask(self, H, W, device):
        if self.shift == 0:
            return None
        img = torch.zeros(1, H, W, 1, device=device)
        slices = (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None))
        cnt = 0
        for h in slices:
            for w in slices:
                img[:, h, w, :] = cnt
                cnt += 1
        win = window_partition(img, self.ws).squeeze(-1)   # (nW, ws*ws)
        mask = win.unsqueeze(2) - win.unsqueeze(1)          # (nW, N, N)
        return mask.masked_fill(mask != 0, -100.).masked_fill(mask == 0, 0.)

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))
        mask = self._make_mask(H, W, x.device)
        wins = window_partition(x, self.ws)
        wins = self.attn(wins, mask)
        x = window_reverse(wins, self.ws, H, W).view(B, L, C)
        if self.shift:
            x = x.view(B, H, W, C)
            x = torch.roll(x, (self.shift, self.shift), dims=(1, 2))
            x = x.view(B, L, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerge(nn.Module):
    """2x spatial downsampling by merging 2x2 patches."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.red  = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                        x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        x = x.view(B, -1, 4 * C)
        return self.red(self.norm(x)), H // 2, W // 2


class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, ws=8, downsample=True, drop_path=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, ws, shift=(i % 2 == 1), drop=drop_path)
            for i in range(depth)
        ])
        self.down = PatchMerge(dim) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        feat = x.view(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
        if self.down:
            x, H, W = self.down(x, H, W)
        return x, H, W, feat   # feat: spatial (B,C,H,W) before downsample


class SwinTBackbone(nn.Module):
    """
    Swin-T backbone producing 4 spatial feature maps at strides [4,8,16,32].
    in_chans=1 for grayscale hologram.
    embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24]  → Swin-T-Small
    embed_dim=96, depths=[2,2,2,2], num_heads=[3,6,12,24]  → Swin-T-Tiny (less VRAM)
    """
    def __init__(self, in_chans=1, embed_dim=96,
                 depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), ws=8):
        super().__init__()
        self.patch = PatchEmbed(in_chans, embed_dim, patch_size=4)
        dims = [embed_dim * (2**i) for i in range(len(depths))]
        self.out_channels = dims          # [96, 192, 384, 768]
        self.stages = nn.ModuleList([
            SwinStage(dims[i], depths[i], num_heads[i], ws,
                      downsample=(i < len(depths)-1))
            for i in range(len(depths))
        ])

    def forward(self, x):
        x, H, W = self.patch(x)
        feats = []
        for stage in self.stages:
            x, H, W, feat = stage(x, H, W)
            feats.append(feat)        # feats[i]: (B, dims[i], H>>i, W>>i)
        return feats                  # 4 feature maps at strides 4,8,16,32


# ──────────────────────────────────────────────────────────────
# FiLM conditioning layer
# ──────────────────────────────────────────────────────────────

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    physics (B,D) → gamma,beta (B,C,1,1) → modulates feature map x (B,C,H,W).
    This is the key mechanism: the network learns the mapping from
    (wavelength, Z, L) to diffraction physics entirely from supervision.
    At inference, physics can be zeroed; the network uses learned priors.
    """
    def __init__(self, physics_dim, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(physics_dim, 128), nn.SiLU(),
            nn.Linear(128, channels * 2),
        )

    def forward(self, x, p):
        g, b = self.fc(p).chunk(2, dim=-1)
        return x * (1 + g.view(g.shape[0], -1, 1, 1)) \
                 + b.view(b.shape[0], -1, 1, 1)


# ──────────────────────────────────────────────────────────────
# YOLO-style PANet neck  (re-implements C2f inline, no import needed)
# ──────────────────────────────────────────────────────────────

class CBS(nn.Module):
    """Conv-BN-SiLU."""
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2), nn.SiLU(inplace=True))
    def forward(self, x): return self.m(x)


class C2f(nn.Module):
    """YOLOv8 C2f block."""
    def __init__(self, c1, c2, n=2):
        super().__init__()
        h = c2 // 2
        self.cv1 = CBS(c1, 2*h, 1, 1, 0)
        self.cv2 = CBS((2+n)*h, c2, 1, 1, 0)
        self.m   = nn.ModuleList(
            [nn.Sequential(CBS(h, h), CBS(h, h)) for _ in range(n)])
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for blk in self.m: y.append(blk(y[-1]))
        return self.cv2(torch.cat(y, 1))


class PANetNeck(nn.Module):
    """
    YOLO-style Path Aggregation Network.
    Top-down (FPN) + bottom-up (PAN) fusion of 4-scale SwinT features.
    """
    def __init__(self, channels):
        super().__init__()
        c0, c1, c2, c3 = channels  # [96,192,384,768]

        # top-down
        self.lat3  = CBS(c3, c2, 1, 1, 0)
        self.td2   = C2f(c2+c2, c2)
        self.lat2  = CBS(c2, c1, 1, 1, 0)
        self.td1   = C2f(c1+c1, c1)
        self.lat1  = CBS(c1, c0, 1, 1, 0)
        self.td0   = C2f(c0+c0, c0)
        # bottom-up
        self.dn0   = CBS(c0, c0, 3, 2)
        self.bu1   = C2f(c0+c1, c1)
        self.dn1   = CBS(c1, c1, 3, 2)
        self.bu2   = C2f(c1+c2, c2)
        self.dn2   = CBS(c2, c2, 3, 2)
        self.bu3   = C2f(c2+c3, c3)
        self.out_channels = [c0, c1, c2, c3]

    def forward(self, fs):
        f0, f1, f2, f3 = fs
        p2 = self.td2(torch.cat([F.interpolate(self.lat3(f3),
                                  f2.shape[-2:], mode='nearest'), f2], 1))
        p1 = self.td1(torch.cat([F.interpolate(self.lat2(p2),
                                  f1.shape[-2:], mode='nearest'), f1], 1))
        p0 = self.td0(torch.cat([F.interpolate(self.lat1(p1),
                                  f0.shape[-2:], mode='nearest'), f0], 1))
        n1 = self.bu1(torch.cat([self.dn0(p0), p1], 1))
        n2 = self.bu2(torch.cat([self.dn1(n1), p2], 1))
        n3 = self.bu3(torch.cat([self.dn2(n2), f3], 1))
        return [p0, n1, n2, n3]


# ──────────────────────────────────────────────────────────────
# FiLM-conditioned phase decoder head
# ──────────────────────────────────────────────────────────────

class FiLMPhaseDecoder(nn.Module):
    """
    Progressive upsampling: stride-32 → full resolution.
    FiLM physics conditioning at every scale.
    Output: (B,1,H,W) phase map in [0,1].
    """
    def __init__(self, neck_ch, physics_dim=4):
        super().__init__()
        c0, c1, c2, c3 = neck_ch

        self.up3  = nn.Sequential(CBS(c3,c2), nn.Upsample(scale_factor=2,
                                    mode='bilinear', align_corners=False))
        self.f3   = FiLM(physics_dim, c2)
        self.fuse3= C2f(c2+c2, c2)

        self.up2  = nn.Sequential(CBS(c2,c1), nn.Upsample(scale_factor=2,
                                    mode='bilinear', align_corners=False))
        self.f2   = FiLM(physics_dim, c1)
        self.fuse2= C2f(c1+c1, c1)

        self.up1  = nn.Sequential(CBS(c1,c0), nn.Upsample(scale_factor=2,
                                    mode='bilinear', align_corners=False))
        self.f1   = FiLM(physics_dim, c0)
        self.fuse1= C2f(c0+c0, c0)

        # stride-4 → stride-1  (two 2x upsamplings)
        self.final = nn.Sequential(
            CBS(c0, 64), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            CBS(64, 32), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, 1), nn.Sigmoid(),
        )

    def forward(self, neck, p):
        n0, n1, n2, n3 = neck
        x = self.f3(self.up3(n3), p);  x = self.fuse3(torch.cat([x, n2], 1))
        x = self.f2(self.up2(x),  p);  x = self.fuse2(torch.cat([x, n1], 1))
        x = self.f1(self.up1(x),  p);  x = self.fuse1(torch.cat([x, n0], 1))
        return self.final(x)            # (B,1,H,W)
