# ultralytics/models/yolo/holography/train.py

import math
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.holography_loss import PhysicsAwareLoss
from ultralytics.nn.modules.holography import (
    SwinTBackbone, PANetNeck, FiLMPhaseDecoder
)


class PhysicsAwareSwinTYOLO(torch.nn.Module):
    """
    Full end-to-end model: SwinTBackbone + PANetNeck + FiLMPhaseDecoder.
    Forward(x, physics) → phase map (B,1,H,W).
    physics can be None at inference (zero-padded, model uses learned priors).
    """
    stride = torch.tensor([32.0])   # satisfies ultralytics stride checks

    def __init__(self, embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24),
                 window_size=8, physics_dim=4):
        super().__init__()
        self.backbone = SwinTBackbone(1, embed_dim, depths, num_heads, window_size)
        ch = self.backbone.out_channels
        self.neck    = PANetNeck(ch)
        self.decoder = FiLMPhaseDecoder(self.neck.out_channels, physics_dim)
        self.physics_dim = physics_dim

    def forward(self, x, physics=None):
        B = x.shape[0]
        if physics is None:
            physics = torch.zeros(B, self.physics_dim, device=x.device, dtype=x.dtype)
        feats      = self.backbone(x)
        neck_feats = self.neck(feats)
        phase      = self.decoder(neck_feats, physics)
        return phase

    def info(self, verbose=True, **kwargs):
        n = sum(p.numel() for p in self.parameters()) / 1e6
        if verbose:
            LOGGER.info(f"PhysicsAwareSwinTYOLO: {n:.1f}M params")
        return n


class HolographyTrainer(BaseTrainer):
    """
    Custom trainer for DLHM phase reconstruction.
    Hooks into BaseTrainer by overriding:
      - get_model       → PhysicsAwareSwinTYOLO
      - get_dataloader  → DLHMDataset / DataLoader
      - preprocess_batch
      - criterion
      - _do_train (loss logging)
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build PhysicsAwareSwinTYOLO from YAML config."""
        mc = cfg if isinstance(cfg, dict) else {}
        model = PhysicsAwareSwinTYOLO(
            embed_dim   = mc.get('embed_dim', 96),
            depths      = tuple(mc.get('depths', [2, 2, 6, 2])),
            num_heads   = tuple(mc.get('num_heads', [3, 6, 12, 24])),
            window_size = mc.get('window_size', 8),
            physics_dim = mc.get('physics_dim', 4),
        )
        if weights:
            ckpt = torch.load(weights, map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            model.load_state_dict(sd, strict=False)
            if verbose:
                LOGGER.info(f"Loaded weights from {weights}")
        self.loss_fn = PhysicsAwareLoss(
            alpha = self.args.get('loss_alpha', 1.0),
            beta  = self.args.get('loss_beta', 1.0),
            gamma = self.args.get('loss_gamma', 0.5),
            delta = self.args.get('loss_delta', 0.5),
        )
        self.loss_names = ['L1', 'SSIM', 'Grad', 'Fourier', 'Total']
        return model

    def get_dataloader(self, dataset_path, batch_size=2, rank=0, mode='train'):
        """Return DLHMDataLoader. dataset_path is the CSV path (passed via data: in YAML)."""
        # Import here to avoid circular imports
        import sys, os
        sys.path.insert(0, str(Path(__file__).parents[5]))  # repo root
        from dlhm_dataset import DLHMDataset
        import pandas as pd

        csv_path = dataset_path
        base_dir = self.args.get('base_dir', '')
        seed     = self.args.get('seed', 42)

        df = pd.read_csv(csv_path)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(df)

        if mode == 'train':
            end = int(n * 0.8)
            df_split = df.iloc[:end]
            augment  = True
            shuffle  = True
        elif mode == 'val':
            start = int(n * 0.8)
            end   = int(n * 0.9)
            df_split = df.iloc[start:end]
            augment  = False
            shuffle  = False
        else:  # test
            start = int(n * 0.9)
            df_split = df.iloc[start:]
            augment  = False
            shuffle  = False

        crop_size = self.args.get('imgsz', 1024)
        ds = DLHMDataset(df_split, base_dir, crop_size=crop_size, augment=augment)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=True,
                          drop_last=(mode == 'train'))

    def preprocess_batch(self, batch):
        """Move tensors to device."""
        device = next(self.model.parameters()).device
        batch['raw']     = batch['raw'].to(device, non_blocking=True)
        batch['gt']      = batch['gt'].to(device, non_blocking=True)
        batch['physics'] = batch['physics'].to(device, non_blocking=True)
        return batch

    def criterion(self, preds, batch):
        """Compute PhysicsAwareLoss."""
        return self.loss_fn(preds, batch['gt'])

    def forward(self, batch):
        """Override BaseTrainer forward to pass physics."""
        return self.model(batch['raw'], batch['physics'])
