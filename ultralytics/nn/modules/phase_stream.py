# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
PhaseYOLO — Phase Texture Stream (Stream B)
FILE: ultralytics/nn/modules/phase_stream.py

WHY THIS EXISTS
---------------
In Digital Holographic Microscopy, oxygenation state of RBCs encodes as
PHASE TEXTURE — the spatial distribution of fringe frequencies and their
symmetry:
  healthy  → regular concentric fringes, 0.08-0.15 cyc/px, symmetric
  normoxia → fringes slightly irregular, mild frequency broadening
  hypoxia  → fringe symmetry broken, dominant 0.15-0.25 cyc/px, spicules

Standard convolutional backbones treat holograms like RGB photos and
have no mechanism to explicitly extract fringe phase statistics.

This module provides TWO components:

1. GaborPyramid
   Three Gabor filter banks at different frequency bands, applied to the
   raw input image. Each bank pools globally (GAP) to produce a fixed-length
   phase descriptor. The three descriptors are concatenated and projected to
   a 128-dim phase embedding.
   → Captures what standard convs cannot: explicit fringe frequency content

2. PhaseGate
   A channel-attention gate that conditions the spatial feature map (from
   Stream A) on the phase embedding. Implemented as a learned linear
   projection from the phase embedding to a per-channel scalar gate.
   → Tells Stream A which channels carry morphology-relevant information
   → Applied at P3, P4, P5 in the backbone (three scales)

PUBLICATION JUSTIFICATION
--------------------------
No existing DHM RBC detection paper uses a dedicated phase-texture stream.
GaborPyramid is physics-motivated (Gabor filters are the canonical model
for V1 simple cells AND for holographic fringe detection). PhaseGate is
a novel fusion mechanism that bridges the physics-informed representation
(phase) with the spatial representation (convolutional features).

INTEGRATION
-----------
Both classes are imported in __init__.py and used via holoYOLOv8n.yaml
as backbone layers (injected before the first Conv stride-2).

The yaml cannot directly express the two-stream dependency, so both
streams are embedded inside a single wrapper: DualStreamStem.
DualStreamStem runs GaborStem (Stream A first layer) AND GaborPyramid
(Stream B) on the same input, stores the phase embedding as an attribute
on the tensor (via a side-channel dict passed through the model), and
returns the spatial features for the rest of the backbone.

PhaseGate is inserted at layers 4, 6, 10 (P3, P4, P5) in the yaml as:
  - [-1, 1, PhaseGate, [64, 128]]   # spatial_ch=64, phase_dim=128
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("GaborPyramid", "PhaseGate", "DualStreamStem", "OrdinalMorphLoss")


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_gabor_kernels(
    base_freqs, n_orient, kernel_size, device, dtype
) -> torch.Tensor:
    """
    Build (n_gabor, 1, k, k) Gabor kernels.
    All maths in float32 for stability, cast to dtype at end.
    """
    k    = kernel_size
    half = k // 2
    yy, xx = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32, device=device),
        torch.arange(-half, half + 1, dtype=torch.float32, device=device),
        indexing="ij",
    )
    n_freq  = len(base_freqs)
    n_gabor = n_freq * n_orient

    freq_vals  = torch.tensor(base_freqs, dtype=torch.float32, device=device)
    freq_vals  = freq_vals.repeat(n_orient)
    theta_vals = torch.linspace(0., math.pi, n_orient + 1)[:-1]
    theta_vals = theta_vals.to(device).repeat_interleave(n_freq)
    sigma      = torch.full((n_gabor,), 3.0, device=device)

    kernels = []
    for i in range(n_gabor):
        cos_t  = torch.cos(theta_vals[i])
        sin_t  = torch.sin(theta_vals[i])
        sx     = sigma[i].clamp(min=0.5)
        x_rot  =  xx * cos_t + yy * sin_t
        y_rot  = -xx * sin_t + yy * cos_t
        gauss  = torch.exp(-0.5 * (x_rot**2 / sx**2 + y_rot**2 / sx**2))
        carrier = torch.cos(2.0 * math.pi * freq_vals[i] * x_rot)
        kernels.append(gauss * carrier)

    return torch.stack(kernels).unsqueeze(1).to(dtype=dtype, device=device)


# ── GaborPyramid ──────────────────────────────────────────────────────────────

class GaborPyramid(nn.Module):
    """
    Three-scale Gabor filter bank applied to the raw hologram.
    Produces a fixed 128-dim phase embedding per image.

    Each scale targets a different fringe frequency band:
      Low  (0.08-0.12 cyc/px): coarse ring pattern, cell boundary
      Mid  (0.12-0.18 cyc/px): biconcave dip region, normoxia signal
      High (0.18-0.25 cyc/px): fine spicule fringes, hypoxia signal

    Each bank: apply filters → Global Average Pool → descriptor
    Three descriptors concatenated → Linear → 128-dim embedding.

    Args:
        in_channels (int): image channels (1 for grayscale hologram)
        out_dim     (int): phase embedding dimension (default 128)
        kernel_size (int): Gabor kernel spatial size (default 15)
        n_orient    (int): orientations per frequency band (default 12)
    """

    BANDS = (
        (0.08, 0.10, 0.12),          # low-freq band
        (0.12, 0.15, 0.18),          # mid-freq band
        (0.18, 0.20, 0.22, 0.25),    # high-freq band (4 freqs for hypoxia)
    )

    def __init__(
        self,
        in_channels: int = 1,
        out_dim:     int = 128,
        kernel_size: int = 15,
        n_orient:    int = 12,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.n_orient    = n_orient
        self.bands       = self.BANDS

        # Learnable parameters for each band
        self.band_params = nn.ModuleList()
        band_dims = []
        for band_freqs in self.bands:
            n_f     = len(band_freqs)
            n_gabor = n_f * n_orient
            band_dim = n_gabor

            # learnable frequency, orientation, sigma per filter
            freq_init  = torch.tensor(band_freqs, dtype=torch.float32).repeat(n_orient)
            theta_init = torch.linspace(0., math.pi, n_orient+1)[:-1].repeat_interleave(n_f)
            params = nn.ParameterDict({
                "frequencies": nn.Parameter(freq_init),
                "thetas":      nn.Parameter(theta_init),
                "sigma":       nn.Parameter(torch.full((n_gabor,), 3.0)),
                "psi":         nn.Parameter(torch.zeros(n_gabor)),
            })
            self.band_params.append(params)
            band_dims.append(band_dim)

        total_dim = sum(band_dims)
        self.proj = nn.Sequential(
            nn.Linear(total_dim, out_dim * 2),
            nn.SiLU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def _gabor_response(self, x: torch.Tensor, params: nn.ParameterDict) -> torch.Tensor:
        """Apply one band's Gabor bank to x, return GAP descriptor."""
        k    = self.kernel_size
        half = k // 2
        dev  = x.device
        dt   = x.dtype

        yy, xx = torch.meshgrid(
            torch.arange(-half, half+1, dtype=torch.float32, device=dev),
            torch.arange(-half, half+1, dtype=torch.float32, device=dev),
            indexing="ij",
        )
        freqs  = params["frequencies"].float()
        thetas = params["thetas"].float()
        sigmas = params["sigma"].float()
        psi    = params["psi"].float()
        n      = len(freqs)

        kernels = []
        for i in range(n):
            cos_t  = torch.cos(thetas[i])
            sin_t  = torch.sin(thetas[i])
            sx     = sigmas[i].abs().clamp(min=0.5)
            x_rot  =  xx * cos_t + yy * sin_t
            y_rot  = -xx * sin_t + yy * cos_t
            gauss  = torch.exp(-0.5 * (x_rot**2 / sx**2 + y_rot**2 / sx**2))
            carrier = torch.cos(2.0 * math.pi * freqs[i] * x_rot + psi[i])
            kernels.append(gauss * carrier)

        W = torch.stack(kernels).unsqueeze(1).to(dtype=dt, device=dev)

        # Apply to each input channel
        responses = []
        for c in range(1):
            xc = x[:, c:c+1]
            responses.append(F.conv2d(xc, W, padding=half))
        feat = torch.cat(responses, dim=1)               # (B, n*in_ch, H, W)
        return feat.mean(dim=(-2, -1))                   # (B, n*in_ch) GAP

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) raw hologram
        Returns:
            phase_emb: (B, out_dim) phase texture embedding
        """
        descriptors = []
        for params in self.band_params:
            descriptors.append(self._gabor_response(x, params))
        concat = torch.cat(descriptors, dim=1)           # (B, total_dim)
        return self.proj(concat)                         # (B, out_dim)


# ── PhaseGate ─────────────────────────────────────────────────────────────────

class PhaseGate(nn.Module):
    """
    Conditions a spatial feature map on a phase embedding via channel attention.

    Given:
      - spatial: (B, spatial_ch, H, W) from the convolutional backbone
      - phase_emb: (B, phase_dim) from GaborPyramid

    Computes a per-channel gate:
      gate = sigmoid(Linear(phase_emb) → (B, spatial_ch, 1, 1))
      output = spatial * (1 + gate)    ← residual: gate refines, not replaces

    This implements a physics-to-spatial information bridge:
    the fringe frequency content (phase_emb) tells the backbone which
    of its spatial channels are carrying morphology-relevant information.

    The phase embedding is stored in a shared dict (`phase_cache`) that
    is attached to the model at build time and populated by DualStreamStem
    at the start of each forward pass.

    Args:
        spatial_ch (int): channels of the incoming spatial feature
        phase_dim  (int): dimension of the phase embedding (default 128)
    """

    def __init__(self, spatial_ch: int, phase_dim: int = 128):
        super().__init__()
        spatial_ch = int(spatial_ch)
        phase_dim  = int(phase_dim)
        self.spatial_ch = spatial_ch

        self.gate = nn.Sequential(
            nn.Linear(phase_dim, spatial_ch * 2),
            nn.SiLU(),
            nn.Linear(spatial_ch * 2, spatial_ch),
        )
        # Cache will be set by the model's DualStreamStem each forward pass
        self.phase_cache: dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase_emb = self.phase_cache.get("emb")
        if phase_emb is None:
            return x  # graceful pass-through if cache not populated
        # Ensure dtype match (AMP)
        if phase_emb.dtype != x.dtype:
            phase_emb = phase_emb.to(dtype=x.dtype)
        g = self.gate(phase_emb)                         # (B, spatial_ch)
        g = torch.sigmoid(g).unsqueeze(-1).unsqueeze(-1) # (B, spatial_ch, 1, 1)
        return x * (1.0 + g)                             # residual gate


# ── DualStreamStem ────────────────────────────────────────────────────────────

class DualStreamStem(nn.Module):
    """
    Entry point of PhaseYOLO — runs both streams on the same input.

    Stream A: GaborStem (physics-informed spatial stem)
      → same as existing GaborStem, outputs (B, out_ch, H/2, W/2)

    Stream B: GaborPyramid (phase texture encoder)
      → outputs (B, phase_dim) phase embedding
      → stores it in a shared dict so PhaseGate modules can access it

    The shared dict is set up by the model's _setup_phase_cache() method
    which is called once after model construction. All PhaseGate layers
    in the model receive a reference to the same dict, so when
    DualStreamStem writes "emb" into it, every PhaseGate reads it.

    Args:
        out_channels (int): GaborStem output channels (default 16)
        in_channels  (int): image channels (default 1, injected by tasks.py)
        phase_dim    (int): phase embedding dimension (default 128)
    """

    def __init__(
        self,
        out_channels: int = 16,
        in_channels:  int = 1,
        phase_dim:    int = 128,
    ):
        super().__init__()
        out_channels = int(out_channels)
        in_channels  = int(in_channels)
        phase_dim    = int(phase_dim)
        mid          = 64

        self.in_channels = in_channels
        self.phase_dim   = phase_dim

        # Stream A — spatial
        from ultralytics.nn.modules.gabor import GaborFilterBank
        self.gabor  = GaborFilterBank(
            in_channels=in_channels,
            out_channels=mid,
            kernel_size=15,
            base_freqs=(0.08, 0.10, 0.12, 0.15, 0.18, 0.22),
            n_orient=12,
        )
        self.dw  = nn.Conv2d(mid, mid, 3, stride=2, padding=1, groups=mid, bias=False)
        self.pw  = nn.Conv2d(mid, out_channels, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        # Stream B — phase texture
        self.phase_pyramid = GaborPyramid(
            in_channels=in_channels,
            out_dim=phase_dim,
            kernel_size=15,
            n_orient=12,
        )

        # Shared cache — populated every forward, read by PhaseGate layers
        self.phase_cache: dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        # Ensure tensor has at least one channel
        if x.shape[1] == 0:
            raise RuntimeError(f"Invalid input with 0 channels: {x.shape}")
    
        # Convert RGB → grayscale
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
    
        # Ensure exactly one channel
        if x.shape[1] != 1:
            x = x[:, :1]
    
        # ---- Stream B ----
        with torch.cuda.amp.autocast(enabled=False):
            phase_emb = self.phase_pyramid(x.float())
    
        self.phase_cache["emb"] = phase_emb
    
        # ---- Stream A ----
        x = self.gabor(x)
        x = self.dw(x)
        x = self.act(self.bn(self.pw(x)))
    
        return x
    
    # ── OrdinalMorphLoss ──────────────────────────────────────────────────────────

class OrdinalMorphLoss(nn.Module):
    """
    Ordinal classification loss for morphological severity ordering.

    RBC oxygenation states have a natural severity ordering:
      healthy (0) < normoxia (1) < hypoxia (2)

    Standard BCE treats all misclassifications equally.
    This loss adds an ordinal penalty proportional to the distance
    between predicted and true class in the severity scale:

      L_ordinal = sum_i max(0, d(pred_i, true_i) - margin) / B

    where d(a, b) = |a - b| (ordinal distance).

    Combined loss:
      L = L_detection (from YOLO) + lambda_ord * L_ordinal

    Use this by subclassing v8DetectionLoss and overriding compute_loss.

    Args:
        nc         (int):   number of classes (3)
        lambda_ord (float): weight of ordinal term (default 0.5)
        margin     (float): ordinal violations within margin not penalised
                            (default 0.0, penalise all violations)
    """

    SEVERITY = {0: 0, 1: 1, 2: 2}  # healthy=0, normoxia=1, hypoxia=2

    def __init__(self, nc: int = 3, lambda_ord: float = 0.5, margin: float = 0.0):
        super().__init__()
        self.nc         = nc
        self.lambda_ord = lambda_ord
        self.margin     = margin

        # Ordinal distance matrix (nc × nc)
        dist = torch.zeros(nc, nc)
        for i in range(nc):
            for j in range(nc):
                dist[i, j] = abs(i - j)
        self.register_buffer("dist_matrix", dist)

    def forward(
        self,
        cls_pred: torch.Tensor,   # (N, nc) raw logits
        cls_true: torch.Tensor,   # (N,)    integer class labels
    ) -> torch.Tensor:
        """
        Args:
            cls_pred: (N, nc) classification logits
            cls_true: (N,)    true class indices (0=healthy,1=normoxia,2=hypoxia)
        Returns:
            ordinal_loss: scalar
        """
        if cls_pred.numel() == 0:
            return cls_pred.sum() * 0.0

        pred_class = cls_pred.argmax(dim=1)              # (N,)
        # Ordinal distance for each prediction
        dist = self.dist_matrix[pred_class, cls_true]    # (N,)
        penalty = torch.clamp(dist - self.margin, min=0.0)
        return self.lambda_ord * penalty.mean()
