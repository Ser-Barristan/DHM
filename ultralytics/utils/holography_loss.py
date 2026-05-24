# ultralytics/utils/holography_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_loss(pred, target, window_size=11, data_range=1.0):
    """Differentiable SSIM loss (1 - SSIM). No external dependency."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    k = window_size
    pad = k // 2

    # Gaussian kernel
    sigma = 1.5
    coords = torch.arange(k, dtype=pred.dtype, device=pred.device) - pad
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)   # (1,1,k,k)

    def apply_blur(x):
        return F.conv2d(x, kernel, padding=pad, groups=x.shape[1])

    mu1 = apply_blur(pred)
    mu2 = apply_blur(target)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    s1  = apply_blur(pred   ** 2) - mu1_sq
    s2  = apply_blur(target ** 2) - mu2_sq
    s12 = apply_blur(pred * target) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return 1.0 - ssim_map.mean()


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, pred, target):
        gx_p = F.conv2d(pred,   self.kx, padding=1)
        gy_p = F.conv2d(pred,   self.ky, padding=1)
        gx_t = F.conv2d(target, self.kx, padding=1)
        gy_t = F.conv2d(target, self.ky, padding=1)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


class FourierLoss(nn.Module):
    """Log-magnitude Fourier domain L1 loss. Physically motivated for holograms."""
    def forward(self, pred, target):
        fp = torch.log1p(torch.abs(torch.fft.fft2(pred)))
        ft = torch.log1p(torch.abs(torch.fft.fft2(target)))
        return F.l1_loss(fp, ft)


class PhysicsAwareLoss(nn.Module):
    """
    L_total = alpha*L1 + beta*(1-SSIM) + gamma*Grad + delta*Fourier

    SSIM is computed inline (no torchmetrics dependency needed in the repo).
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, delta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.grad_loss    = GradientLoss()
        self.fourier_loss = FourierLoss()

    def forward(self, pred, target):
        l1   = F.l1_loss(pred, target)
        ss   = ssim_loss(pred, target)
        grad = self.grad_loss(pred, target)
        four = self.fourier_loss(pred, target)
        total = self.alpha * l1 + self.beta * ss + self.gamma * grad + self.delta * four
        return total, {'l1': l1.item(), 'ssim_loss': ss.item(),
                       'grad': grad.item(), 'fourier': four.item()}
