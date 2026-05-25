# ultralytics/models/yolo/holography/val.py
import math
import torch
import torch.nn.functional as F
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER


def _ssim_val(pred, target, ws=11):
    C1, C2 = 0.01**2, 0.03**2
    pad = ws // 2
    coords = torch.arange(ws, dtype=pred.dtype, device=pred.device) - pad
    g = torch.exp(-coords**2 / (2*1.5**2)); g /= g.sum()
    k = g.outer(g).view(1,1,ws,ws)
    def blur(x): return F.conv2d(x, k, padding=pad)
    mu1,mu2 = blur(pred), blur(target)
    s1  = blur(pred**2)    - mu1**2
    s2  = blur(target**2)  - mu2**2
    s12 = blur(pred*target) - mu1*mu2
    return ((2*mu1*mu2+C1)*(2*s12+C2) /
            ((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean().item()


class HolographyValidator(BaseValidator):
    """
    Validator for phase reconstruction.
    Computes: MAE, MSE, PSNR, SSIM, Phase-RMSE, Phase-RMSE-rad.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args=args, _callbacks=_callbacks)
        self._reset()

    def _reset(self):
        self._acc = dict(mae=0, mse=0, psnr=0, ssim=0,
                         phase_rmse=0, phase_rmse_rad=0)
        self._n = 0

    def preprocess(self, batch):
        dev = self.device
        batch['raw']     = batch['raw'].to(dev)
        batch['gt']      = batch['gt'].to(dev)
        batch['physics'] = batch['physics'].to(dev)
        return batch

    def postprocess(self, preds):
        return preds   # (B,1,H,W) already

    def init_metrics(self, model):
        self._reset()

    def update_metrics(self, preds, batch):
        p = preds.float()
        t = batch['gt'].float()
        B = p.shape[0]

        mse  = torch.mean((p - t)**2).item()
        mae  = torch.mean(torch.abs(p - t)).item()
        psnr = 10 * math.log10(1.0 / (mse + 1e-10))
        ssim = _ssim_val(p, t)
        rmse = math.sqrt(mse)

        self._acc['mae']           += mae  * B
        self._acc['mse']           += mse  * B
        self._acc['psnr']          += psnr * B
        self._acc['ssim']          += ssim * B
        self._acc['phase_rmse']    += rmse * B
        self._acc['phase_rmse_rad']+= rmse * 2 * math.pi * B
        self._n += B

    def get_stats(self):
        if self._n == 0:
            return self._acc
        return {k: v / self._n for k, v in self._acc.items()}

    def print_results(self):
        s = self.get_stats()
        LOGGER.info(
            f"  Phase Reconstruction Metrics:\n"
            f"  MAE={s['mae']:.4f}  MSE={s['mse']:.6f}  "
            f"PSNR={s['psnr']:.2f}dB  SSIM={s['ssim']:.4f}\n"
            f"  Phase-RMSE={s['phase_rmse']:.4f}  "
            f"Phase-RMSE-rad={s['phase_rmse_rad']:.4f}"
        )
        return s

    # BaseValidator expects self.metrics.results_dict for CSV saving
    @property
    def metrics(self):
        class _M:
            def __init__(self, stats):
                self.results_dict = stats
                self.fitness = stats.get('ssim', 0.0)  # best model selected by SSIM
        return _M(self.get_stats())
