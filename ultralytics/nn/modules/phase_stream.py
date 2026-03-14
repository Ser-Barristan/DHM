# Ultralytics YOLO 🚀, AGPL-3.0 license
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("GaborPyramid", "PhaseGate", "DualStreamStem", "OrdinalMorphLoss")


# ─────────────────────────────────────────────────────────────
# Gabor Pyramid
# ─────────────────────────────────────────────────────────────

class GaborPyramid(nn.Module):

    BANDS = (
        (0.08, 0.10, 0.12),
        (0.12, 0.15, 0.18),
        (0.18, 0.20, 0.22, 0.25),
    )

    def __init__(self, in_channels=1, out_dim=128, kernel_size=15, n_orient=12):
        super().__init__()

        self.kernel_size = kernel_size
        self.n_orient = n_orient

        self.band_params = nn.ModuleList()
        band_dims = []

        for band_freqs in self.BANDS:
            n_f = len(band_freqs)
            n_gabor = n_f * n_orient
            band_dim = n_gabor

            freq_init = torch.tensor(band_freqs).repeat(n_orient)
            theta_init = torch.linspace(0., math.pi, n_orient + 1)[:-1].repeat_interleave(n_f)

            params = nn.ParameterDict({
                "frequencies": nn.Parameter(freq_init),
                "thetas": nn.Parameter(theta_init),
                "sigma": nn.Parameter(torch.full((n_gabor,), 3.0)),
                "psi": nn.Parameter(torch.zeros(n_gabor)),
            })

            self.band_params.append(params)
            band_dims.append(band_dim)

        total_dim = sum(band_dims)

        self.proj = nn.Sequential(
            nn.Linear(total_dim, out_dim * 2),
            nn.SiLU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def _gabor_response(self, x, params):

        k = self.kernel_size
        half = k // 2
        dev = x.device

        yy, xx = torch.meshgrid(
            torch.arange(-half, half + 1, device=dev),
            torch.arange(-half, half + 1, device=dev),
            indexing="ij",
        )

        freqs = params["frequencies"]
        thetas = params["thetas"]
        sigmas = params["sigma"]
        psi = params["psi"]

        kernels = []

        for i in range(len(freqs)):

            cos_t = torch.cos(thetas[i])
            sin_t = torch.sin(thetas[i])

            x_rot = xx * cos_t + yy * sin_t
            y_rot = -xx * sin_t + yy * cos_t

            gauss = torch.exp(-0.5 * (x_rot**2 + y_rot**2) / (sigmas[i]**2 + 1e-6))
            carrier = torch.cos(2 * math.pi * freqs[i] * x_rot + psi[i])

            kernels.append(gauss * carrier)

        W = torch.stack(kernels).unsqueeze(1)

        feat = F.conv2d(x, W, padding=half)
        return feat.mean(dim=(-2, -1))

    def forward(self, x):

        descriptors = []

        for params in self.band_params:
            descriptors.append(self._gabor_response(x, params))

        concat = torch.cat(descriptors, dim=1)
        return self.proj(concat)


# ─────────────────────────────────────────────────────────────
# Phase Gate
# ─────────────────────────────────────────────────────────────

class PhaseGate(nn.Module):

    def __init__(self, spatial_ch, phase_dim=128):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(phase_dim, spatial_ch * 2),
            nn.SiLU(),
            nn.Linear(spatial_ch * 2, spatial_ch),
        )

        self.phase_cache = {}

    def forward(self, x):

        phase_emb = self.phase_cache.get("emb")

        if phase_emb is None:
            return x

        if phase_emb.dtype != x.dtype:
            phase_emb = phase_emb.to(x.dtype)

        g = torch.sigmoid(self.gate(phase_emb)).unsqueeze(-1).unsqueeze(-1)

        return x * (1 + g)


# ─────────────────────────────────────────────────────────────
# Dual Stream Stem  (FIXED)
# ─────────────────────────────────────────────────────────────

class DualStreamStem(nn.Module):

    def __init__(self, out_channels=16, in_channels=1, phase_dim=128):

        super().__init__()

        from ultralytics.nn.modules.gabor import GaborFilterBank

        mid = 64

        self.gabor = GaborFilterBank(
            in_channels=1,
            out_channels=mid,
            kernel_size=15,
            base_freqs=(0.08,0.10,0.12,0.15,0.18,0.22),
            n_orient=12,
        )

        self.dw = nn.Conv2d(mid, mid, 3, stride=2, padding=1, groups=mid, bias=False)
        self.pw = nn.Conv2d(mid, out_channels, 1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        self.phase_pyramid = GaborPyramid(1, phase_dim)

        self.phase_cache = {}

    def forward(self, x):

        # --------- CRITICAL FIX ---------
        # Guarantee tensor shape is (B,1,H,W)

        if x.ndim == 3:
            x = x.unsqueeze(1)

        if x.ndim != 4:
            raise RuntimeError(f"Invalid tensor shape {x.shape}")

        # If tensor has zero channels (augmentation bug), repair it
        if x.shape[1] == 0:
            x = torch.zeros(x.shape[0],1,x.shape[2],x.shape[3], device=x.device)

        # Convert RGB → grayscale
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        # Guarantee exactly 1 channel
        if x.shape[1] != 1:
            x = x[:,0:1,:,:]

        # --------- Phase stream ---------

        with torch.cuda.amp.autocast(enabled=False):
            phase_emb = self.phase_pyramid(x.float())

        self.phase_cache["emb"] = phase_emb

        # --------- Spatial stream ---------

        x = self.gabor(x)
        x = self.dw(x)
        x = self.act(self.bn(self.pw(x)))

        return x


# ─────────────────────────────────────────────────────────────
# Ordinal Loss
# ─────────────────────────────────────────────────────────────

class OrdinalMorphLoss(nn.Module):

    def __init__(self, nc=3, lambda_ord=0.5, margin=0.0):
        super().__init__()

        self.lambda_ord = lambda_ord
        self.margin = margin

        dist = torch.zeros(nc, nc)

        for i in range(nc):
            for j in range(nc):
                dist[i,j] = abs(i-j)

        self.register_buffer("dist_matrix", dist)

    def forward(self, cls_pred, cls_true):

        if cls_pred.numel()==0:
            return cls_pred.sum()*0.0

        pred_class = cls_pred.argmax(dim=1)

        dist = self.dist_matrix[pred_class, cls_true]

        penalty = torch.clamp(dist - self.margin, min=0.0)

        return self.lambda_ord * penalty.mean()
