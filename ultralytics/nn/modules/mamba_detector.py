import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------
# 1.  SELECTIVE SSM  (compact, stable Mamba-style core)
# ----------------------------------------------------------
class MambaSSM(nn.Module):
    """
    Compact selective SSM block.
    Keeps the Mamba flavor, but avoids the huge projection that was exploding params.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = max(4, d_model // 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )

        # dt_rank + B + C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            A.log().unsqueeze(0).expand(self.d_inner, -1).clone()
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _ssm_scan(self, u, delta, A, B, C):
        """
        u     : (B, L, D)
        delta : (B, L, D)
        A     : (D, N)
        B,C   : (B, L, N)
        """
        B_b, L, D = u.shape
        N = A.shape[1]

        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))   # (B,L,D,N)
        dB = delta.unsqueeze(-1) * B.unsqueeze(2)                            # (B,L,D,N)

        h = torch.zeros(B_b, D, N, device=u.device, dtype=u.dtype)
        ys = []

        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))  # (B,D)

        return torch.stack(ys, dim=1)  # (B,L,D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model) -> (B, L, d_model)
        """
        B, L, _ = x.shape

        xz = self.in_proj(x)                  # (B,L,2*d_inner)
        xv, z = xz.chunk(2, dim=-1)           # (B,L,d_inner)

        xv = self.conv1d(xv.transpose(1, 2))[:, :, :L].transpose(1, 2)
        xv = F.silu(xv)

        x_params = self.x_proj(xv)            # (B,L,dt_rank + 2*d_state)

        dt_raw = x_params[:, :, :self.dt_rank]
        bc = x_params[:, :, self.dt_rank:self.dt_rank + 2 * self.d_state]
        B_p, C_p = bc.chunk(2, dim=-1)        # each (B,L,d_state)

        dt = F.softplus(self.dt_proj(dt_raw)) # (B,L,d_inner)

        A = -torch.exp(self.A_log)            # (d_inner, d_state)
        y = self._ssm_scan(xv, dt, A, B_p, C_p)

        y = y + xv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return y


# ----------------------------------------------------------
# 2.  MAMBA BLOCK
# ----------------------------------------------------------
class MambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ssm = MambaSSM(dim, d_state=d_state)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ----------------------------------------------------------
# 3.  PATCH EMBED STEM
# ----------------------------------------------------------
class MambaStem(nn.Module):
    def __init__(self, in_ch: int = 3, embed_dim: int = 64, patch_sz: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_sz, stride=patch_sz)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        return self.norm(tokens), H, W


# ----------------------------------------------------------
# 4.  PATCH MERGE
# ----------------------------------------------------------
class _PatchMerge(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.proj = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat(
            [x[:, 0::2, 0::2],
             x[:, 1::2, 0::2],
             x[:, 0::2, 1::2],
             x[:, 1::2, 1::2]],
            dim=-1
        )
        x = x.view(B, -1, 4 * C)
        return self.proj(self.norm(x)), H // 2, W // 2


# ----------------------------------------------------------
# 5.  MAMBA STAGE
# ----------------------------------------------------------
class MambaStage(nn.Module):
    def __init__(self, dim: int, depth: int, d_state: int = 16, downsample: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([MambaBlock(dim, d_state) for _ in range(depth)])
        self.down = _PatchMerge(dim) if downsample else None

    def forward(self, x, H, W):
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
    Outputs:
        P2: e
        P3: 2e
        P4: 4e
        P5: 4e
    """
    def __init__(self, in_ch: int = 3, embed_dim: int = 64,
                 depths: tuple = (2, 2, 4, 2), d_state: int = 16):
        super().__init__()
        d0, d1, d2, d3 = depths
        e = embed_dim

        self.stem = MambaStem(in_ch, e, patch_sz=4)
        self.stage1 = MambaStage(e,     d0, d_state, downsample=True)
        self.stage2 = MambaStage(e * 2, d1, d_state, downsample=True)
        self.stage3 = MambaStage(e * 4, d2, d_state, downsample=True)
        self.proj3 = nn.Linear(e * 8, e * 4, bias=False)
        self.stage4 = MambaStage(e * 4, d3, d_state, downsample=False)

        self.out_channels = [e, e * 2, e * 4, e * 4]

    def forward(self, x: torch.Tensor):
        tokens, H, W = self.stem(x)
        P2, tokens, H, W = self.stage1(tokens, H, W)
        P3, tokens, H, W = self.stage2(tokens, H, W)
        P4, tokens, H, W = self.stage3(tokens, H, W)
        tokens = self.proj3(tokens)
        P5, _, _, _ = self.stage4(tokens, H, W)
        return [P2, P3, P4, P5]


# ----------------------------------------------------------
# 7.  AGASPP
# ----------------------------------------------------------
class _ChannelGate(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hid = max(ch // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).view(x.shape[0], -1, 1, 1)
        return x * w


class AGASPPDet(nn.Module):
    def __init__(self, in_ch: int = 256, out_ch: int = 256, rates: tuple = (1, 3, 6, 9)):
        super().__init__()
        mid = out_ch // 2

        self.branches = nn.ModuleList()
        self.gates = nn.ModuleList()
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, padding=r, dilation=r, bias=False),
                nn.GroupNorm(8, mid),
                nn.SiLU(inplace=True),
            ))
            self.gates.append(_ChannelGate(mid))

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.GroupNorm(8, mid),
            nn.SiLU(inplace=True),
        )
        self.gap_gate = _ChannelGate(mid)

        self.proj = nn.Sequential(
            nn.Conv2d(mid * (len(rates) + 1), out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        parts = [g(b(x)) for b, g in zip(self.branches, self.gates)]
        gap_f = self.gap_gate(self.gap(x))
        gap_f = F.interpolate(gap_f, size=(H, W), mode="bilinear", align_corners=False)
        parts.append(gap_f)
        return self.proj(torch.cat(parts, dim=1))


# ----------------------------------------------------------
# 8.  3-LEVEL BiFPN (P3/P4/P5 only)
# ----------------------------------------------------------
class _BiFPNNode(nn.Module):
    def __init__(self, ch: int, n_in: int = 2):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_in, dtype=torch.float32))
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.GroupNorm(8, ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, *tensors):
        w = F.relu(self.w)
        w = w / (w.sum() + 1e-4)
        fused = sum(wi * t for wi, t in zip(w, tensors))
        return self.act(self.bn(self.pw(self.dw(fused))))


class BiFPNDet(nn.Module):
    def __init__(self, in_dims: tuple, neck_ch: int = 256, num_rounds: int = 3):
        super().__init__()
        c3, c4, c5 = in_dims

        def _lat(cin):
            return nn.Sequential(
                nn.Conv2d(cin, neck_ch, 1, bias=False),
                nn.GroupNorm(8, neck_ch),
                nn.SiLU(inplace=True),
            )

        self.lat_p3 = _lat(c3)
        self.lat_p4 = _lat(c4)
        self.lat_p5 = _lat(c5)

        self.rounds = nn.ModuleList()
        for _ in range(num_rounds):
            self.rounds.append(nn.ModuleDict({
                "td_p4": _BiFPNNode(neck_ch, n_in=2),
                "td_p3": _BiFPNNode(neck_ch, n_in=2),
                "bu_p4": _BiFPNNode(neck_ch, n_in=3),
                "bu_p5": _BiFPNNode(neck_ch, n_in=2),
            }))

    @staticmethod
    def _up(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _down(x, size):
        return F.adaptive_avg_pool2d(x, size)

    def forward(self, features: list) -> list:
        P3, P4, P5 = features

        P3 = self.lat_p3(P3)
        P4 = self.lat_p4(P4)
        P5 = self.lat_p5(P5)

        for rnd in self.rounds:
            P4_td = rnd["td_p4"](P4, self._up(P5, P4.shape[-2:]))
            P3_out = rnd["td_p3"](P3, self._up(P4_td, P3.shape[-2:]))

            P4_out = rnd["bu_p4"](P4, P4_td, self._down(P3_out, P4_td.shape[-2:]))
            P5_out = rnd["bu_p5"](P5, self._down(P4_out, P5.shape[-2:]))

            P3, P4, P5 = P3_out, P4_out, P5_out

        return [P3, P4, P5]


# ----------------------------------------------------------
# 9.  SPPF -> AGASPP -> BiFPN
# ----------------------------------------------------------
class MambaSPPFASPPBiFPNNeck(nn.Module):
    def __init__(self,
                 backbone_dims: tuple = (128, 256, 256),
                 neck_ch: int = 256,
                 sppf_k: int = 5,
                 aspp_rates: tuple = (1, 3, 6, 9),
                 bifpn_rounds: int = 3):
        super().__init__()
        c3, c4, c5 = backbone_dims

        c5_mid = c5 // 2
        self.sppf_cv1 = nn.Sequential(
            nn.Conv2d(c5, c5_mid, 1, bias=False),
            nn.GroupNorm(8, c5_mid),
            nn.SiLU(inplace=True),
        )
        self.sppf_pool = nn.MaxPool2d(kernel_size=sppf_k, stride=1, padding=sppf_k // 2)
        self.sppf_cv2 = nn.Sequential(
            nn.Conv2d(c5_mid * 4, c5, 1, bias=False),
            nn.GroupNorm(8, c5),
            nn.SiLU(inplace=True),
        )

        self.agaspp = AGASPPDet(in_ch=c5, out_ch=neck_ch, rates=aspp_rates)
        self.bifpn = BiFPNDet(in_dims=(c3, c4, neck_ch), neck_ch=neck_ch, num_rounds=bifpn_rounds)

    def _sppf(self, x):
        y = self.sppf_cv1(x)
        y1 = self.sppf_pool(y)
        y2 = self.sppf_pool(y1)
        y3 = self.sppf_pool(y2)
        return self.sppf_cv2(torch.cat([y, y1, y2, y3], dim=1))

    def forward(self, backbone_feats: list) -> list:
        _, P3, P4, P5 = backbone_feats
        P5 = self._sppf(P5)
        P5 = self.agaspp(P5)
        return self.bifpn([P3, P4, P5])


# ----------------------------------------------------------
# 10. FULL DETECTOR
# ----------------------------------------------------------
class MambaSPPFASPPBiFPNDetector(nn.Module):
    def __init__(self,
                 in_ch: int = 3,
                 embed_dim: int = 64,
                 depths: tuple = (2, 2, 4, 2),
                 d_state: int = 16,
                 neck_ch: int = 256,
                 sppf_k: int = 5,
                 aspp_rates: tuple = (1, 3, 6, 9),
                 bifpn_rounds: int = 3):
        super().__init__()

        self.backbone = MambaBackboneNet(
            in_ch=in_ch,
            embed_dim=embed_dim,
            depths=depths,
            d_state=d_state,
        )

        e = embed_dim
        c3 = e * 2
        c4 = e * 4
        c5 = e * 4

        self.neck = MambaSPPFASPPBiFPNNeck(
            backbone_dims=(c3, c4, c5),
            neck_ch=neck_ch,
            sppf_k=sppf_k,
            aspp_rates=aspp_rates,
            bifpn_rounds=bifpn_rounds,
        )

        self.out_channels = [neck_ch, neck_ch, neck_ch]

    def forward(self, x: torch.Tensor) -> list:
        feats = self.backbone(x)
        return self.neck(feats)
