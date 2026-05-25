# ultralytics/utils/holography_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ssim(pred, target, ws=11):
    """Pure-torch differentiable SSIM. Returns 1-SSIM as loss scalar."""
    C1, C2 = 0.01**2, 0.03**2
    pad = ws // 2
    coords = torch.arange(ws, dtype=pred.dtype, device=pred.device) - pad
    g = torch.exp(-coords**2 / (2 * 1.5**2))
    g = g / g.sum()
    k = g.outer(g).view(1, 1, ws, ws)

    def blur(x): return F.conv2d(x, k, padding=pad, groups=1)

    mu1, mu2     = blur(pred), blur(target)
    s1  = blur(pred**2)      - mu1**2
    s2  = blur(target**2)    - mu2**2
    s12 = blur(pred*target)  - mu1*mu2

    num = (2*mu1*mu2 + C1) * (2*s12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return 1.0 - (num / den).mean()


class GradLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                           dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                           dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, p, t):
        return (F.l1_loss(F.conv2d(p, self.kx, padding=1),
                          F.conv2d(t, self.kx, padding=1)) +
                F.l1_loss(F.conv2d(p, self.ky, padding=1),
                          F.conv2d(t, self.ky, padding=1)))


class FourierLoss(nn.Module):
    """Log-magnitude FFT L1 loss. Physically motivated for holographic fringes."""
    def forward(self, p, t):
        lp = torch.log1p(torch.abs(torch.fft.fft2(p)))
        lt = torch.log1p(torch.abs(torch.fft.fft2(t)))
        return F.l1_loss(lp, lt)


class PhysicsAwareLoss(nn.Module):
    """
    L = alpha*L1 + beta*(1-SSIM) + gamma*GradL1 + delta*FourierL1
    All weights configurable via YAML / overrides dict.
    Returns: (total_loss_tensor, loss_items_tensor[4])
    loss_items format matches BaseTrainer expectation: 1-D tensor of floats.
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, delta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.grad = GradLoss()
        self.four = FourierLoss()

    def forward(self, pred, target):
        l1   = F.l1_loss(pred, target)
        ss   = _ssim(pred, target)
        gr   = self.grad(pred, target)
        fo   = self.four(pred, target)
        tot  = self.alpha*l1 + self.beta*ss + self.gamma*gr + self.delta*fo
        # loss_items: detached 1-D tensor for BaseTrainer logging (l1, ssim, grad, fourier)
        items = torch.tensor([l1.item(), ss.item(), gr.item(), fo.item()],
                              device=pred.device)
        return tot, items
