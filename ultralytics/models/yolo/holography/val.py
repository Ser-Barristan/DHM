# ultralytics/models/yolo/holography/val.py

import math
import torch
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER


def compute_ssim_simple(pred, target, window_size=11):
    """Pure-torch SSIM for validation metrics."""
    import torch.nn.functional as F
    C1, C2 = 0.01**2, 0.03**2
    k, pad = window_size, window_size // 2
    sigma = 1.5
    coords = torch.arange(k, dtype=pred.dtype, device=pred.device) - pad
    g = torch.exp(-coords**2 / (2*sigma**2)); g = g / g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)
    def blur(x): return F.conv2d(x, kernel, padding=pad, groups=1)
    mu1,mu2 = blur(pred), blur(target)
    s1  = blur(pred**2) - mu1**2
    s2  = blur(target**2) - mu2**2
    s12 = blur(pred*target) - mu1*mu2
    ssim_map = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))
    return ssim_map.mean().item()


class HolographyValidator(BaseValidator):
    """Validator for phase reconstruction metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_accum = {k: 0.0 for k in
            ['mae', 'mse', 'psnr', 'ssim', 'phase_rmse', 'phase_rmse_rad']}
        self.n = 0

    def preprocess(self, batch):
        device = self.device
        batch['raw']     = batch['raw'].to(device)
        batch['gt']      = batch['gt'].to(device)
        batch['physics'] = batch['physics'].to(device)
        return batch

    def postprocess(self, preds):
        return preds

    def update_metrics(self, preds, batch):
        pred   = preds.float()
        target = batch['gt'].float()
        B = pred.shape[0]

        mae  = torch.mean(torch.abs(pred - target)).item()
        mse  = torch.mean((pred - target)**2).item()
        psnr = 10 * math.log10(1.0 / (mse + 1e-10))
        ssim = compute_ssim_simple(pred, target)
        ph_rmse = math.sqrt(mse)
        ph_rmse_rad = ph_rmse * 2 * math.pi

        for k, v in zip(
            ['mae', 'mse', 'psnr', 'ssim', 'phase_rmse', 'phase_rmse_rad'],
            [mae, mse, psnr, ssim, ph_rmse, ph_rmse_rad]
        ):
            self.metrics_accum[k] += v * B
        self.n += B

    def get_stats(self):
        if self.n == 0:
            return self.metrics_accum
        return {k: v / self.n for k, v in self.metrics_accum.items()}

    def print_results(self):
        stats = self.get_stats()
        LOGGER.info(
            f"  MAE={stats['mae']:.4f}  MSE={stats['mse']:.6f}  "
            f"PSNR={stats['psnr']:.2f}dB  SSIM={stats['ssim']:.4f}  "
            f"PhRMSE={stats['phase_rmse']:.4f}  PhRMSE_rad={stats['phase_rmse_rad']:.4f}"
        )
        return stats
