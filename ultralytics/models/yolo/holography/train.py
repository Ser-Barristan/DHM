# ultralytics/models/yolo/holography/train.py
from copy import copy
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.holography_loss import PhysicsAwareLoss
from ultralytics.nn.modules.holography import (
    SwinTBackbone, PANetNeck, FiLMPhaseDecoder
)


# ── Full model assembled here so HolographyTrainer.get_model can return it ──

class PhysicsAwareSwinTYOLO(nn.Module):
    """
    SwinTBackbone → PANetNeck → FiLMPhaseDecoder.
    forward(x, physics) → phase (B,1,H,W) in [0,1].
    physics=None at inference → zeros (learned priors activate).
    stride attribute required by BaseTrainer stride checks.
    """
    stride = torch.tensor([4., 8., 16., 32.])  # multi-scale strides

    def __init__(self, embed_dim=96, depths=(2,2,6,2),
                 num_heads=(3,6,12,24), ws=8, physics_dim=4):
        super().__init__()
        self.backbone = SwinTBackbone(1, embed_dim, depths, num_heads, ws)
        self.neck     = PANetNeck(self.backbone.out_channels)
        self.head     = FiLMPhaseDecoder(self.neck.out_channels, physics_dim)
        self.physics_dim = physics_dim

    def forward(self, x, physics=None):
        if physics is None:
            physics = torch.zeros(x.shape[0], self.physics_dim,
                                  device=x.device, dtype=x.dtype)
        return self.head(self.neck(self.backbone(x)), physics)

    def info(self, verbose=True, **kw):
        n = sum(p.numel() for p in self.parameters()) / 1e6
        if verbose:
            LOGGER.info(f'PhysicsAwareSwinTYOLO  {n:.1f}M params')
        return n


# ── Trainer ──────────────────────────────────────────────────────────────────

class HolographyTrainer(BaseTrainer):
    """
    Subclass of BaseTrainer for DLHM holographic phase reconstruction.

    Overrides required by BaseTrainer._do_train loop:
      get_model          → builds PhysicsAwareSwinTYOLO
      get_dataloader     → returns DLHMDataset DataLoader
      preprocess_batch   → moves tensors to device
      criterion          → PhysicsAwareLoss(pred, batch['gt'])
      get_validator      → HolographyValidator
      label_loss_items   → names the 4 loss components
      progress_string    → custom tqdm display string
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build model from YAML cfg dict or defaults."""
        c = cfg if isinstance(cfg, dict) else {}
        model = PhysicsAwareSwinTYOLO(
            embed_dim   = c.get('embed_dim',  96),
            depths      = tuple(c.get('depths',    [2, 2, 6, 2])),
            num_heads   = tuple(c.get('num_heads',  [3, 6, 12, 24])),
            ws          = c.get('window_size', 8),
            physics_dim = c.get('physics_dim', 4),
        )
        if weights:
            sd = torch.load(weights, map_location='cpu')
            sd = sd.get('model_state_dict', sd.get('state_dict', sd))
            model.load_state_dict(sd, strict=False)
            if verbose:
                LOGGER.info(f'Loaded weights: {weights}')

        # PhysicsAwareLoss — store on trainer so criterion() can use it
        self.loss_fn = PhysicsAwareLoss(
            alpha = float(self.args.get('loss_alpha', 1.0)),
            beta  = float(self.args.get('loss_beta',  1.0)),
            gamma = float(self.args.get('loss_gamma', 0.5)),
            delta = float(self.args.get('loss_delta', 0.5)),
        )
        self.loss_names = ['L1', 'SSIM', 'Grad', 'Fourier']
        model.info(verbose)
        return model

    # BaseTrainer calls get_dataloader('train_data_path', batch, rank, mode)
    # where train_data_path comes from args.data (set in overrides)
    def get_dataloader(self, dataset_path, batch_size=2, rank=0, mode='train'):
        import sys, os
        # dlhm_dataset.py lives at repo root (DHM/)
        repo_root = str(Path(__file__).parents[4])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from dlhm_dataset import DLHMDataset

        csv_path = dataset_path              # args.data → CSV path
        base_dir = str(self.args.get('base_dir', ''))
        seed     = int(self.args.get('seed', 42))
        imgsz    = int(self.args.get('imgsz', 1024))

        df = pd.read_csv(csv_path).sample(frac=1, random_state=seed).reset_index(drop=True)
        n  = len(df)
        n_train, n_val = int(n * 0.8), int(n * 0.1)

        splits = {'train': df.iloc[:n_train],
                  'val':   df.iloc[n_train:n_train + n_val],
                  'test':  df.iloc[n_train + n_val:]}
        augment = (mode == 'train')
        shuffle = (mode == 'train')

        ds = DLHMDataset(splits[mode], base_dir, crop_size=imgsz, augment=augment)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=True,
                          drop_last=(mode == 'train'))

    def preprocess_batch(self, batch):
        """Move all batch tensors to the training device."""
        dev = next(self.model.parameters()).device
        batch['raw']     = batch['raw'].to(dev, non_blocking=True)
        batch['gt']      = batch['gt'].to(dev, non_blocking=True)
        batch['physics'] = batch['physics'].to(dev, non_blocking=True)
        return batch

    def criterion(self, preds, batch):
        """
        Called by BaseTrainer._do_train as:
            self.loss, self.loss_items = self.criterion(preds, batch)
        Must return (loss_tensor, loss_items_1d_tensor).
        """
        return self.loss_fn(preds, batch['gt'])

    # BaseTrainer._do_train calls model(batch) internally as:
    #   preds = self.model(batch)   ← this won't pass physics!
    # We override _do_train's model call by wrapping in forward:
    def _model_forward(self, batch):
        """Custom forward used inside _do_train to pass physics."""
        return self.model(batch['raw'], batch['physics'])

    def get_validator(self):
        from ultralytics.models.yolo.holography.val import HolographyValidator
        self.loss_names = ['L1', 'SSIM', 'Grad', 'Fourier']
        return HolographyValidator(
            self.test_loader, save_dir=self.save_dir,
            args=copy(self.args), _callbacks=self.callbacks,
        )

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        BaseTrainer calls this to build the CSV header and progress string.
        Must return dict when loss_items given, list of keys otherwise.
        """
        keys = [f'{prefix}/{n}' for n in self.loss_names]
        if loss_items is not None:
            return {k: round(float(v), 5)
                    for k, v in zip(keys, loss_items)}
        return keys

    def progress_string(self):
        """Custom tqdm header matching label_loss_items keys."""
        return (('\n' + '%11s' * (4 + len(self.loss_names))) %
                ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size'))
