"""
WaveRingLoss — novel composite loss for ring-structured object detection.
File location: ultralytics/utils/loss_wavering.py  (NEW FILE)

Drop-in alongside the existing loss.py.  Reference it in the Trainer override
(see train_waveyolo.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import v8DetectionLoss          # ultralytics built-in base
from ..nn.modules.waveyolo import HaarWavelet2D


# ──────────────────────────────────────────────────────────────────────────────
# Component 1 — RingIoU
# ──────────────────────────────────────────────────────────────────────────────

def ring_iou_loss(pred: torch.Tensor, target: torch.Tensor,
                  rho: float = 0.45) -> torch.Tensor:
    """IoU-based loss that accounts for the annular shape of targets.

    On top of CIoU it penalises centroid displacement relative to outer
    radius (ring objects whose centres are shifted look wrong even at high
    standard IoU) and adds a soft aspect-ratio-unity term (rings ≈ circles).

    Args:
        pred   : (N, 4) predicted boxes  [cx, cy, w, h]  normalised
        target : (N, 4) ground-truth boxes  [cx, cy, w, h]  normalised
        rho    : inner/outer radius ratio (tune per dataset; 0.45 is generic)

    Returns:
        Scalar loss tensor.
    """
    # ── CIoU base ──────────────────────────────────────────────────────
    pw, ph = pred[:, 2].clamp(1e-6), pred[:, 3].clamp(1e-6)
    tw, th = target[:, 2].clamp(1e-6), target[:, 3].clamp(1e-6)

    # Intersection
    inter_w = (torch.min(pred[:, 0] + pw / 2, target[:, 0] + tw / 2)
               - torch.max(pred[:, 0] - pw / 2, target[:, 0] - tw / 2)).clamp(0)
    inter_h = (torch.min(pred[:, 1] + ph / 2, target[:, 1] + th / 2)
               - torch.max(pred[:, 1] - ph / 2, target[:, 1] - th / 2)).clamp(0)
    inter   = inter_w * inter_h
    union   = pw * ph + tw * th - inter + 1e-7
    iou     = inter / union

    # Enclosing diagonal² and centre distance²
    enclose_w = (torch.max(pred[:, 0] + pw / 2, target[:, 0] + tw / 2)
                 - torch.min(pred[:, 0] - pw / 2, target[:, 0] - tw / 2)).clamp(1e-7)
    enclose_h = (torch.max(pred[:, 1] + ph / 2, target[:, 1] + th / 2)
                 - torch.min(pred[:, 1] - ph / 2, target[:, 1] - th / 2)).clamp(1e-7)
    c2        = enclose_w ** 2 + enclose_h ** 2 + 1e-7
    rho2      = ((pred[:, 0] - target[:, 0]) ** 2
                 + (pred[:, 1] - target[:, 1]) ** 2)
    v         = (4 / (torch.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha_v = v / (1 - iou + v + 1e-7)
    ciou = iou - rho2 / c2 - alpha_v * v

    # ── Ring centroid penalty ───────────────────────────────────────────
    # Outer radius of GT box (mean of semi-axes)
    r_outer = (tw / 2 + th / 2) / 2.
    dc      = ((pred[:, 0] - target[:, 0]) ** 2
               + (pred[:, 1] - target[:, 1]) ** 2).sqrt()
    ring_pen = (dc / (r_outer + 1e-7)).clamp(0., 1.)

    # ── Physics prior: aspect ratio → 1 (rings ≈ circles) ─────────────
    ar_loss = (pw / (ph + 1e-7) - 1.).abs()

    loss = (1. - ciou) + 0.3 * ring_pen + 0.1 * ar_loss
    return loss.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Component 2 — WaveletEdgeLoss
# ──────────────────────────────────────────────────────────────────────────────

class WaveletEdgeLoss(nn.Module):
    """Penalise blurry ring boundary predictions via HF wavelet energy.

    Compares the HH (diagonal high-frequency) sub-band between predicted
    and ground-truth segmentation / heatmap masks.  Only active when the
    model outputs a mask branch (e.g. YOLO-Seg).  If no mask is available,
    returns 0 and can be disabled by setting lam_wave=0.

    Args:
        eps : stability epsilon
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.haar = HaarWavelet2D()
        self.eps  = eps

    def forward(self, pred_mask: torch.Tensor,
                gt_mask: torch.Tensor) -> torch.Tensor:
        """
        pred_mask, gt_mask : (B, 1, H, W) float tensors in [0, 1]
        """
        if pred_mask is None or gt_mask is None:
            return torch.tensor(0., device='cpu')
        _, _, _, HH_pred = self.haar(pred_mask)
        _, _, _, HH_gt   = self.haar(gt_mask)
        return F.mse_loss(HH_pred, HH_gt)


# ──────────────────────────────────────────────────────────────────────────────
# Component 3 — SNR-weighted Varifocal Loss
# ──────────────────────────────────────────────────────────────────────────────

class SNRWeightedVFL(nn.Module):
    """Varifocal classification loss with per-image SNR-based weighting.

    In low-SNR microscopy images the annotation quality and feature
    discriminability are inherently lower.  Down-weighting those images
    reduces noisy gradient contributions without discarding them entirely.

    Args:
        gamma : focusing exponent (same role as focal loss γ; default 2.0)
        snr_norm : SNR value at which weight == 1.0 (default 5.0)
    """

    def __init__(self, gamma: float = 2.0, snr_norm: float = 5.0):
        super().__init__()
        self.gamma    = gamma
        self.snr_norm = snr_norm

    @staticmethod
    def _estimate_snr(img: torch.Tensor) -> torch.Tensor:
        """Fast SNR proxy: mean / std over spatial dims, then batch-mean."""
        mu    = img.mean(dim=[-2, -1], keepdim=True)
        sigma = img.std(dim=[-2, -1], keepdim=True).clamp(min=1e-6)
        return (mu / sigma).mean()

    def forward(self, pred_logits: torch.Tensor,
                targets: torch.Tensor,
                img: torch.Tensor) -> torch.Tensor:
        """
        pred_logits : (N,)  raw logits
        targets     : (N,)  soft labels in [0,1]  (varifocal-style)
        img         : (B, C, H, W) raw input images for SNR estimation
        """
        snr    = self._estimate_snr(img).clamp(0.5, 15.)
        w      = (snr / self.snr_norm).clamp(0.2, 1.0)

        p = torch.sigmoid(pred_logits)
        q = targets.float()

        # Varifocal loss
        vfl = torch.where(
            q > 0,
            q  * (q - p) ** self.gamma * F.logsigmoid(pred_logits),
            p ** self.gamma * F.logsigmoid(-pred_logits),
        )
        return -(w * vfl).mean()


# ──────────────────────────────────────────────────────────────────────────────
# WaveRingLoss — full composite loss
# ──────────────────────────────────────────────────────────────────────────────

class WaveRingLoss(nn.Module):
    """Composite loss:

        L = λ1·RingIoU  +  λ2·WaveletEdge  +  λ3·SNR-VFL  +  λ4·AspectRatio

    λ defaults are tuned for a single-class ring detection task.  Adjust
    lam_wave to 0 if your model has no mask branch.

    Args:
        rho      : inner/outer radius ratio for RingIoU  (default 0.45)
        lam_ring : weight for RingIoU loss               (default 1.5)
        lam_wave : weight for WaveletEdge loss           (default 0.3)
        lam_cls  : weight for SNR-VFL cls loss           (default 1.0)
        lam_phys : weight for aspect-ratio prior         (default 0.2)
        gamma    : VFL focusing exponent                 (default 2.0)
    """

    def __init__(self,
                 rho: float      = 0.45,
                 lam_ring: float = 1.5,
                 lam_wave: float = 0.3,
                 lam_cls: float  = 1.0,
                 lam_phys: float = 0.2,
                 gamma: float    = 2.0):
        super().__init__()
        self.rho      = rho
        self.lam_ring = lam_ring
        self.lam_wave = lam_wave
        self.lam_cls  = lam_cls
        self.lam_phys = lam_phys

        self.wave_edge = WaveletEdgeLoss()
        self.snr_vfl   = SNRWeightedVFL(gamma=gamma)

    def forward(self,
                pred_box:   torch.Tensor,
                gt_box:     torch.Tensor,
                pred_cls:   torch.Tensor,
                gt_cls:     torch.Tensor,
                img:        torch.Tensor,
                pred_mask:  torch.Tensor = None,
                gt_mask:    torch.Tensor = None) -> dict:
        """
        Returns a dict with 'loss' (total) and individual components
        for logging / ablation tables.
        """
        l_ring = ring_iou_loss(pred_box, gt_box, rho=self.rho)
        l_wave = self.wave_edge(pred_mask, gt_mask)
        l_cls  = self.snr_vfl(pred_cls, gt_cls, img)

        # Physics prior (aspect ratio toward 1)
        pw = pred_box[:, 2].clamp(1e-6)
        ph = pred_box[:, 3].clamp(1e-6)
        l_phys = (pw / (ph + 1e-7) - 1.).abs().mean()

        total = (self.lam_ring * l_ring
                 + self.lam_wave * l_wave
                 + self.lam_cls  * l_cls
                 + self.lam_phys * l_phys)

        return {
            'loss':      total,
            'ring_iou':  l_ring.detach(),
            'wave_edge': l_wave.detach() if isinstance(l_wave, torch.Tensor) else l_wave,
            'snr_vfl':   l_cls.detach(),
            'phys':      l_phys.detach(),
        }
