# dlhm_dataset.py  — place at DHM/ repo root
import os, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image

STATS = {
    'wavelength': (0.405, 0.654),
    'L':          (18.33, 18.96),
    'z':          (0.435, 2.441),
}

def _norm(v, key):
    lo, hi = STATS[key]
    return (v - lo) / (hi - lo + 1e-8)

def _remap(win_path, base_dir):
    p = win_path.replace('\\', '/')
    m = 'MW-Dataset/'
    i = p.find(m)
    rel = p[i+len(m):] if i != -1 else os.path.basename(p)
    return os.path.join(base_dir, rel)

def _load_gray(path):
    img = Image.open(path).convert('L')
    t = torch.from_numpy(np.array(img, np.float32) / 255.0)
    return t.unsqueeze(0)                        # (1,H,W)

def _load_phase(path):
    img = Image.open(path)
    arr = np.array(img, np.float32)
    mn, mx = arr.min(), arr.max()
    arr = (arr - mn) / (mx - mn + 1e-8) if mx > mn else arr * 0.0
    t = torch.from_numpy(arr)
    return (t.unsqueeze(0) if t.ndim == 2 else t[0:1])  # (1,H,W)

def _crop(t, sz):
    _, h, w = t.shape
    return t[:, (h-sz)//2:(h-sz)//2+sz, (w-sz)//2:(w-sz)//2+sz]


class DLHMDataset(Dataset):
    def __init__(self, df, base_dir, crop_size=1024, augment=False, pixel_size=3.8):
        self.df        = df.reset_index(drop=True)
        self.base_dir  = base_dir
        self.crop_size = crop_size
        self.augment   = augment
        self.pixel_size= pixel_size

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw = _crop(_load_gray( _remap(row['raw_full_path'],    self.base_dir)), self.crop_size)
        gt  = _crop(_load_phase(_remap(row['phase_image_path'], self.base_dir)), self.crop_size)

        if self.augment:
            if random.random() > 0.5: raw, gt = TF.hflip(raw), TF.hflip(gt)
            if random.random() > 0.5: raw, gt = TF.vflip(raw), TF.vflip(gt)
            k = random.choice([0,1,2,3])
            if k:
                raw = torch.rot90(raw, k, [-2,-1])
                gt  = torch.rot90(gt,  k, [-2,-1])
            noise   = torch.randn_like(raw) * 0.01
            raw     = torch.clamp(raw + noise, 0, 1)
            speckle = torch.randn_like(raw) * 0.02
            raw     = torch.clamp(raw + raw*speckle, 0, 1)

        physics = torch.tensor([
            _norm(float(row['Wavelength']), 'wavelength'),
            _norm(float(row['L_value']),    'L'),
            _norm(float(row['z_value']),    'z'),
            self.pixel_size / 10.0,          # ~0.38, fixed for all samples
        ], dtype=torch.float32)

        return {'raw': raw, 'gt': gt, 'physics': physics,
                'wavelength': float(row['Wavelength']),
                'L': float(row['L_value']), 'z': float(row['z_value'])}
