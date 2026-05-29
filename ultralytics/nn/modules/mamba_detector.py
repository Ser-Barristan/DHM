import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------
# 1.  SELECTIVE SSM  (Mamba core)
# ----------------------------------------------------------
class MambaSSM(nn.Module):
    """
    Simplified Selective State-Space Model (Mamba-style).

    Args:
        d_model  : token / channel dimension
        d_state  : SSM state dimension  (default 16)
        d_conv   : local depthwise conv width  (default 4)
        expand   : inner expansion ratio  (default 2)
    """
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand

        # input split projection: x-branch + z-gate
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)

        # causal local conv on x-branch  (groups=d_inner → depthwise)
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True)

        # selective parameters Δ, B, C from x
        self.x_proj = nn.Linear(
            d_inner,
            2 * d_inner * d_state + d_inner,
            bias=False
        )
        self.dt_proj  = nn.Linear(d_inner, d_inner, bias=True)

        # fixed A matrix (log-space), skip-connection D
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log    = nn.Parameter(
            A.log().unsqueeze(0).expand(d_inner, -1).clone())
        self.D        = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    # ---- SSM scan  (sequential over L, fast for short seqs) ----
    def _ssm_scan(self, u, delta, A, B, C):
        """
        u      : (B, L, d_inner)
        delta  : (B, L, d_inner)
        A      : (d_inner, d_state)   fixed
        B      : (B, L, d_state)
        C      : (B, L, d_state)
        Returns y: (B, L, d_inner)
        """
        B_b, L, D = u.shape
        N          = A.shape[1]
        # discretise A, B
        dA = torch.exp(
            delta.unsqueeze(-1) *                  # (B,L,D,1)
            A.unsqueeze(0).unsqueeze(0))            # (1,1,D,N)
        dB = (delta.unsqueeze(-1) *                 # (B,L,D,1)
              B.unsqueeze(2))                       # (B,L,1,N)  → (B,L,D,N)

        h  = torch.zeros(B_b, D, N,
                         device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h  = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
            ys.append((h * C[:, t].unsqueeze(2)).sum(-1))   # (B,D)
        return torch.stack(ys, dim=1)               # (B,L,D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)  →  (B, L, d_model)"""
        residual = x
        B, L, _  = x.shape

        # 1. project → xv (active) + z (gate)
        xz       = self.in_proj(x)                 # (B,L,2*d_inner)
        xv, z    = xz.chunk(2, dim=-1)             # each (B,L,d_inner)

        # 2. causal local conv (truncate to L to remove future padding)
        xv = self.conv1d(
            xv.transpose(1, 2))[:, :, :L].transpose(1, 2)
        xv = F.silu(xv)

        # 3. selective params
        xBCdt = self.x_proj(xv)                    # (B,L, 2N+d_inner)
        d_state = self.A_log.shape[1]
        bc = xBCdt[:, :, : 2*d_inner*d_state]
        
        B_p, C_p = bc.chunk(2, dim=-1)
        
        B_p = B_p.view(B, L, d_inner, d_state)
        
        C_p = C_p.view(B, L, d_inner, d_state)
        dt   = F.softplus(
            self.dt_proj(xBCdt[:, :, 2 * d_state:]))

        # 4. SSM
        A    = -torch.exp(self.A_log)               # (d_inner, N)
        y    = self._ssm_scan(xv, dt, A, B_p, C_p) # (B,L,d_inner)
        y    = y + xv * self.D.unsqueeze(0).unsqueeze(0)

        # 5. gated output
        y    = y * F.silu(z)
        out  = self.out_proj(y)
        return self.norm(out + residual)


# ----------------------------------------------------------
# 2.  MAMBA BLOCK  (SSM + MLP)
# ----------------------------------------------------------
class MambaBlock(nn.Module):
    """
    One Mamba block: pre-norm SSM + pre-norm MLP with residuals.

    Args:
        dim     : channel dimension
        d_state : SSM state size  (default 16)
    """
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ssm   = MambaSSM(dim, d_state)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """x: (B, H*W, dim)  →  (B, H*W, dim)"""
        x = self.ssm(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x


# ----------------------------------------------------------
# 3.  PATCH EMBED STEM  (Conv2d, stride = patch_size)
# ----------------------------------------------------------
class MambaStem(nn.Module):
    """
    4× downsampling patch embedding.

    Args:
        in_ch     : input channels (3 for RGB)
        embed_dim : output token dimension
        patch_sz  : stride / kernel (default 4)
    """
    def __init__(self, in_ch: int = 3, embed_dim: int = 64,
                 patch_sz: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_sz, stride=patch_sz)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        """Returns tokens (B, H*W, C), H, W"""
        x        = self.proj(x)                    # (B,C,H,W)
        B, C, H, W = x.shape
        tokens   = x.flatten(2).transpose(1, 2)    # (B, H*W, C)
        return self.norm(tokens), H, W


# ----------------------------------------------------------
# 4.  PATCH MERGING  (2× spatial downsample inside backbone)
# ----------------------------------------------------------
class _PatchMerge(nn.Module):
    """Pixel-shuffle concat → linear: (B,L,C) → (B,L/4,2C)"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.proj = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2],
                        x[:, 1::2, 0::2],
                        x[:, 0::2, 1::2],
                        x[:, 1::2, 1::2]], dim=-1)   # (B,H/2,W/2,4C)
        x = x.view(B, -1, 4 * C)
        return self.proj(self.norm(x)), H // 2, W // 2


# ----------------------------------------------------------
# 5.  MAMBA STAGE  (N MambaBlocks + optional PatchMerge)
# ----------------------------------------------------------
class MambaStage(nn.Module):
    """
    One backbone stage.

    Args:
        dim       : input token dimension
        depth     : number of MambaBlock layers
        d_state   : SSM state size
        downsample: if True, append PatchMerge (doubles channels)
    """
    def __init__(self, dim: int, depth: int,
                 d_state: int = 16, downsample: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MambaBlock(dim, d_state) for _ in range(depth)])
        self.down   = _PatchMerge(dim) if downsample else None

    def forward(self, x, H, W):
        """
        Returns:
            feat_map : (B, C, H, W)  feature map at this resolution
            x_next   : tokens for next stage  (B, L', C') or same if no down
            H', W'   : spatial dims for next stage
        """
        for blk in self.blocks:
            x = blk(x, H, W)

        B, _, C = x.shape
        feat_map = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        if self.down is not None:
            x_next, H_next, W_next = self.down(x, H, W)
        else:
            x_next, H_next, W_next = x, H, W

        return feat_map, x_next, H_next, W_next


# ----------------------------------------------------------
# 6.  FULL MAMBA BACKBONE
# ----------------------------------------------------------
class MambaBackboneNet(nn.Module):
    """
    Hierarchical Mamba backbone for detection.

    Stage outputs (feature maps before downsampling):
        P2 : stride 4   dim = embed_dim          (128×128 @ 512 input)
        P3 : stride 8   dim = embed_dim * 2       (64×64)
        P4 : stride 16  dim = embed_dim * 4       (32×32)
        P5 : stride 32  dim = embed_dim * 4 (*)   (16×16)

    (*) Stage 4 uses no downsample; channel count stays at 4×embed_dim
        because Stage 3's PatchMerge already doubled 2× → 4×.

    Args:
        in_ch     : input image channels  (3)
        embed_dim : base channel width    (64)
        depths    : (d1,d2,d3,d4) MambaBlock depth per stage
        d_state   : SSM state size
    """
    def __init__(self, in_ch: int = 3, embed_dim: int = 64,
                 depths: tuple = (2, 2, 4, 2),
                 d_state: int = 16):
        super().__init__()
        d0, d1, d2, d3 = depths
        e = embed_dim

        self.stem    = MambaStem(in_ch, e, patch_sz=4)

        # Stage 1: dim=e,    output P2 (stride 4),  then merge → 2e
        self.stage1  = MambaStage(e,     d0, d_state, downsample=True)
        # Stage 2: dim=2e,   output P3 (stride 8),  then merge → 4e
        self.stage2  = MambaStage(e * 2, d1, d_state, downsample=True)
        # Stage 3: dim=4e,   output P4 (stride 16), then merge → 8e… but we cap
        # We cap at 4e by using a projection after merge
        self.stage3  = MambaStage(e * 4, d2, d_state, downsample=True)
        # After stage3 merge we have 8e; project back to 4e to cap params
        self.proj3   = nn.Linear(e * 8, e * 4, bias=False)
        # Stage 4: dim=4e,   output P5 (stride 32), NO downsample
        self.stage4  = MambaStage(e * 4, d3, d_state, downsample=False)

        # output channel counts for each stage
        self.out_channels = [e, e * 2, e * 4, e * 4]  # P2..P5

    def forward(self, x: torch.Tensor):
        """
        x : (B, 3, H, W)
        Returns list [P2, P3, P4, P5] of (B, C, H', W') feature maps.
        """
        tokens, H, W = self.stem(x)              # stride 4

        # Stage 1  →  P2 (stride 4)
        P2, tokens, H, W = self.stage1(tokens, H, W)

        # Stage 2  →  P3 (stride 8)
        P3, tokens, H, W = self.stage2(tokens, H, W)

        # Stage 3  →  P4 (stride 16)
        P4, tokens, H, W = self.stage3(tokens, H, W)
        # tokens now has dim 8e; project down to 4e
        tokens = self.proj3(tokens)

        # Stage 4  →  P5 (stride 32)
        P5, _, _, _ = self.stage4(tokens, H, W)

        return [P2, P3, P4, P5]


# ----------------------------------------------------------
# 7.  AGASPP  (Attention-Gated ASPP on P5)
# ----------------------------------------------------------
class _ChannelGate(nn.Module):
    """Squeeze-and-Excite channel attention."""
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hid = max(ch // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, hid),  nn.ReLU(inplace=True),
            nn.Linear(hid, ch),  nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).view(x.shape[0], -1, 1, 1)
        return x * w


class AGASPPDet(nn.Module):
    """
    Attention-Gated Atrous Spatial Pyramid Pooling.

    All branches: 1×1 + 3×3 dilated with rates (1, 6, 12, 18) + GAP.
    Each branch output is gated by a channel attention module.
    Fused features are projected to `out_ch`.

    Args:
        in_ch   : input channels   (256 from backbone P5)
        out_ch  : output channels  (256)
        rates   : dilation rates   (1, 6, 12, 18)
    """
    def __init__(self, in_ch: int = 256, out_ch: int = 256,
                 rates: tuple = (1, 6, 12, 18)):
        super().__init__()
        mid = out_ch // 2          # 128 intermediate to save params

        self.branches = nn.ModuleList()
        self.gates    = nn.ModuleList()
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, mid, 3,
                          padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(mid), nn.SiLU(inplace=True),
            ))
            self.gates.append(_ChannelGate(mid))

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU(inplace=True),
        )
        self.gap_gate = _ChannelGate(mid)

        total_branches = len(rates) + 1             # 5
        self.proj = nn.Sequential(
            nn.Conv2d(mid * total_branches, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W   = x.shape[-2:]
        parts  = [g(b(x)) for b, g in zip(self.branches, self.gates)]
        gap_f  = self.gap_gate(self.gap(x))
        gap_f  = F.interpolate(gap_f, size=(H, W),
                               mode='bilinear', align_corners=False)
        parts.append(gap_f)
        return self.proj(torch.cat(parts, dim=1))


# ----------------------------------------------------------
# 8.  BiFPN NECK  (multi-round, 3-level: P3/P4/P5)
# ----------------------------------------------------------
class _BiFPNNode(nn.Module):
    """
    One weighted-fusion BiFPN node.
    Combines `n_in` (2 or 3) tensors with learned fast-norm weights.
    """
    def __init__(self, ch: int, n_in: int = 2):
        super().__init__()
        self.w    = nn.Parameter(torch.ones(n_in, dtype=torch.float32))
        # depthwise-separable conv
        self.dw   = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw   = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn   = nn.BatchNorm2d(ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, *tensors):
        w = F.relu(self.w)
        w = w / (w.sum() + 1e-4)
        fused = sum(wi * t for wi, t in zip(w, tensors))
        return self.act(self.bn(self.pw(self.dw(fused))))


class BiFPNDet(nn.Module):
    """
    Bidirectional Feature Pyramid Network for 3 levels (P3, P4, P5).

    Accepts features with potentially different input channels and
    projects everything to `neck_ch` on the first call.  All subsequent
    rounds work at `neck_ch`.

    Topology per round
    ------------------
    TOP-DOWN path (P5 → P4 → P3):
        P4_td  = node(P4,  up(P5))          [2-input]
        P3_out = node(P3,  up(P4_td))       [2-input]

    BOTTOM-UP path (P3 → P4 → P5):
        P4_out = node(P4,  P4_td, down(P3_out))   [3-input]
        P5_out = node(P5,  down(P4_out))           [2-input]

    Args:
        in_dims  : (c_p3, c_p4, c_p5) input channel counts
        neck_ch  : unified neck channel width  (256)
        num_rounds : number of BiFPN rounds    (3)
    """
    def __init__(self, in_dims: tuple, neck_ch: int = 256,
                 num_rounds: int = 3):
        super().__init__()
        c3, c4, c5 = in_dims

        # --- lateral projections (input → neck_ch) ---
        def _lat(cin):
            return nn.Sequential(
                nn.Conv2d(cin, neck_ch, 1, bias=False),
                nn.BatchNorm2d(neck_ch), nn.SiLU(inplace=True),
            )
        self.lat_p3 = _lat(c3)
        self.lat_p4 = _lat(c4)
        self.lat_p5 = _lat(c5)

        # --- BiFPN rounds ---
        self.rounds = nn.ModuleList()
        for _ in range(num_rounds):
            self.rounds.append(nn.ModuleDict({
                # top-down
                'td_p4': _BiFPNNode(neck_ch, n_in=2),
                'td_p3': _BiFPNNode(neck_ch, n_in=2),
                # bottom-up
                'bu_p4': _BiFPNNode(neck_ch, n_in=3),
                'bu_p5': _BiFPNNode(neck_ch, n_in=2),
            }))

        self.neck_ch = neck_ch

    @staticmethod
    def _up(x, size):
        return F.interpolate(x, size=size,
                             mode='bilinear', align_corners=False)

    @staticmethod
    def _down(x, size):
        return F.adaptive_avg_pool2d(x, size)

    def forward(self, features: list) -> list:
        """
        features : [P3, P4, P5]  (from backbone + AGASPP)
        Returns  : [P3_out, P4_out, P5_out]  all at neck_ch
        """
        P3, P4, P5 = features

        # lateral projection
        P3 = self.lat_p3(P3)
        P4 = self.lat_p4(P4)
        P5 = self.lat_p5(P5)

        for rnd in self.rounds:
            # --- TOP-DOWN ---
            # P4_td fuses P4 + up(P5) to P4 size
            P4_td  = rnd['td_p4'](P4, self._up(P5, P4.shape[-2:]))
            # P3_out fuses P3 + up(P4_td) to P3 size
            P3_out = rnd['td_p3'](P3, self._up(P4_td, P3.shape[-2:]))

            # --- BOTTOM-UP ---
            # P4_out fuses P4 + P4_td + down(P3_out) to P4 size
            P4_out = rnd['bu_p4'](
                P4, P4_td,
                self._down(P3_out, P4_td.shape[-2:]))
            # P5_out fuses P5 + down(P4_out) to P5 size
            P5_out = rnd['bu_p5'](
                P5, self._down(P4_out, P5.shape[-2:]))

            # carry updated feature maps into next round
            P3 = P3_out
            P4 = P4_out
            P5 = P5_out

        return [P3, P4, P5]


# ----------------------------------------------------------
# 9.  MONOLITHIC NECK MODULE: SPPF → AGASPP → BiFPN
#     (wraps all three so the YAML can reference a single class)
# ----------------------------------------------------------
class MambaSPPFASPPBiFPNNeck(nn.Module):
    """
    Full neck: SPPF on P5  →  AGASPP on P5  →  BiFPN(P3,P4,P5).

    Args:
        backbone_dims : (c_p3, c_p4, c_p5) channels from backbone
                        e.g. (128, 256, 256) for embed_dim=64
        neck_ch       : unified BiFPN channel width (256)
        sppf_k        : SPPF pool kernel size (5)
        aspp_rates    : dilation rates for AGASPP
        bifpn_rounds  : number of BiFPN rounds
    """
    def __init__(self,
                 backbone_dims: tuple = (128, 256, 256),
                 neck_ch: int = 256,
                 sppf_k: int = 5,
                 aspp_rates: tuple = (1, 6, 12, 18),
                 bifpn_rounds: int = 3):
        super().__init__()
        c3, c4, c5 = backbone_dims

        # --- SPPF on P5 ---
        c5_mid = c5 // 2
        self.sppf_cv1 = nn.Sequential(
            nn.Conv2d(c5, c5_mid, 1, bias=False),
            nn.BatchNorm2d(c5_mid), nn.SiLU(inplace=True))
        self.sppf_pool = nn.MaxPool2d(
            kernel_size=sppf_k, stride=1, padding=sppf_k // 2)
        self.sppf_cv2 = nn.Sequential(
            nn.Conv2d(c5_mid * 4, c5, 1, bias=False),
            nn.BatchNorm2d(c5), nn.SiLU(inplace=True))

        # --- AGASPP on P5 (after SPPF) ---
        self.agaspp = AGASPPDet(
            in_ch=c5, out_ch=neck_ch, rates=aspp_rates)

        # --- BiFPN: P3 unchanged, P4 unchanged, P5 after agaspp ---
        self.bifpn = BiFPNDet(
            in_dims=(c3, c4, neck_ch),
            neck_ch=neck_ch,
            num_rounds=bifpn_rounds)

        self.neck_ch = neck_ch

    def _sppf(self, x):
        """SPPF forward."""
        y  = self.sppf_cv1(x)
        y1 = self.sppf_pool(y)
        y2 = self.sppf_pool(y1)
        y3 = self.sppf_pool(y2)
        return self.sppf_cv2(torch.cat([y, y1, y2, y3], dim=1))

    def forward(self, backbone_feats: list) -> list:
        """
        backbone_feats : [P2, P3, P4, P5]
        Returns        : [P3_out, P4_out, P5_out]  each (B, neck_ch, H, W)
        """
        _, P3, P4, P5 = backbone_feats   # P2 not used in detection neck

        # SPPF + AGASPP on P5
        P5 = self._sppf(P5)
        P5 = self.agaspp(P5)

        # BiFPN fusion
        return self.bifpn([P3, P4, P5])


# ----------------------------------------------------------
# 10. FULL END-TO-END MODEL
#     (backbone + neck in one nn.Module so the YAML can call it
#      as a single block; the Detect head is wired in the YAML)
# ----------------------------------------------------------
class MambaSPPFASPPBiFPNDetector(nn.Module):
    """
    Full backbone + neck for the Mamba-SPPF-AGASPP-BiFPN pipeline.

    Returns a list of three feature maps [P3, P4, P5] that feed
    directly into the Ultralytics Detect head.

    Args:
        in_ch        : input image channels       (3)
        embed_dim    : Mamba base channel width   (64)
        depths       : Mamba block counts         (2,2,4,2)
        d_state      : SSM state size             (16)
        neck_ch      : BiFPN / AGASPP channels    (256)
        sppf_k       : SPPF pool kernel           (5)
        aspp_rates   : AGASPP dilation rates      (1,6,12,18)
        bifpn_rounds : BiFPN repetitions          (3)
    """
    def __init__(self,
                 in_ch: int = 3,
                 embed_dim: int = 64,
                 depths: tuple = (2, 2, 4, 2),
                 d_state: int = 16,
                 neck_ch: int = 256,
                 sppf_k: int = 5,
                 aspp_rates: tuple = (1, 6, 12, 18),
                 bifpn_rounds: int = 3):
        super().__init__()

        # ---- backbone ----
        self.backbone = MambaBackboneNet(
            in_ch=in_ch,
            embed_dim=embed_dim,
            depths=depths,
            d_state=d_state,
        )
        # backbone output dims: [e, 2e, 4e, 4e]
        e  = embed_dim
        c3 = e * 2    # P3 channels
        c4 = e * 4    # P4 channels
        c5 = e * 4    # P5 channels

        # ---- neck ----
        self.neck = MambaSPPFASPPBiFPNNeck(
            backbone_dims=(c3, c4, c5),
            neck_ch=neck_ch,
            sppf_k=sppf_k,
            aspp_rates=aspp_rates,
            bifpn_rounds=bifpn_rounds,
        )

        self.out_channels = [neck_ch, neck_ch, neck_ch]

    def forward(self, x: torch.Tensor) -> list:
        """
        x : (B, 3, H, W)
        Returns [P3_out, P4_out, P5_out]  strides [8, 16, 32]
        """
        feats = self.backbone(x)
        return self.neck(feats)
