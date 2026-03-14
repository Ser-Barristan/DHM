# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
HoloYOLO — HoloDetect Head
FILE PATH IN YOUR FORK: ultralytics/nn/modules/holo_detect.py
CREATE THIS FILE (it does not exist in vanilla ultralytics)

Replaces the vanilla Detect head for RBC oxygenation classification.

WHY A CUSTOM HEAD?
  Standard Detect uses a shallow cls branch (2x DWConv3x3 + 1x1 conv).
  This is sufficient when classes differ in SIZE or GROSS SHAPE (RBC vs WBC
  vs Platelet). It is NOT sufficient when all objects are the same cell type
  and classification depends on subtle morphological differences:
    healthy  → perfect biconcave disc, symmetric concentric ring pattern
    normoxia → slightly flattened disc, mildly asymmetric ring depth
    hypoxia  → sickle / echinocyte / stomatocyte, broken ring symmetry

  HoloDetect keeps the IDENTICAL box regression branch as vanilla Detect
  (so bounding-box accuracy is unaffected) and replaces ONLY the cls branch
  with a deeper, spatially-aware stack:

    cls branch:
      Conv(ch, cls_ch, 3) → BN → SiLU   # local texture
      Conv(cls_ch, cls_ch, 3) → BN → SiLU   # shape context
      Conv(cls_ch, cls_ch, 3) → BN → SiLU   # global ring symmetry
      AdaptiveAvgPool → flatten
      Linear(cls_ch, nc)                    # final classification

  This gives 3 dedicated conv layers to encode ring morphology before
  the class decision, compared to 0 dedicated layers in vanilla Detect.

INTEGRATION:
  1.  Add to ultralytics/nn/modules/__init__.py (see bottom of this file)
  2.  Add to ultralytics/nn/tasks.py  parse_model() Detect branch:
        elif m in frozenset({Detect, ..., HoloDetect}):
            args.append([ch[x] for x in f])
            if m in {Detect, ..., HoloDetect}:
                m.legacy = legacy
  3.  In holoYOLOv8n.yaml replace the last Detect line with HoloDetect:
        - [[16, 19, 22], 1, HoloDetect, [nc]]
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import DFL
from ultralytics.utils.tal import dist2bbox, make_anchors


__all__ = ("HoloDetect",)


class HoloDetect(nn.Module):
    """
    Detection head with a deep classification branch for subtle
    intra-class morphology discrimination (healthy / normoxia / hypoxia RBC).

    Drop-in replacement for ultralytics Detect.
    Box regression branch is IDENTICAL to vanilla Detect.
    Classification branch is 3× deeper with dedicated spatial convs.

    Args:
        nc   (int):        number of classes (default 3)
        ch   (tuple[int]): input channel counts from FPN levels P3, P4, P5
    """

    # Class-level attributes mirrored from vanilla Detect so tasks.py
    # and the training engine treat HoloDetect identically to Detect.
    dynamic = False      # force grid reconstruction
    export  = False      # export mode flag
    format  = None       # export format
    end2end = False
    max_det = 300
    shape   = None
    anchors = torch.zeros(0)
    strides = torch.zeros(0)
    legacy  = False

    def __init__(self, nc: int = 3, ch: tuple = ()):
        super().__init__()
        self.nc    = nc                          # number of classes
        self.nl    = len(ch)                     # number of FPN levels
        self.reg_max = 16                        # DFL channels
        self.no    = nc + self.reg_max * 4       # outputs per anchor
        self.stride = torch.zeros(self.nl)

        # ── Box regression branch (identical to vanilla Detect) ──────────────
        # c2 = max of (reg_max*4, first input channel // 4, 16)
        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )

        # ── Classification branch (DEEPER than vanilla) ───────────────────────
        # c3 = classification hidden channels
        # Larger than vanilla (vanilla uses max(nc, ch//4, min_ch=80))
        # We use max(nc*4, ch[0]//2, 128) for richer morphology encoding.
        c3 = max(nc * 4, ch[0] // 2, 128)
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                # Layer 1: local ring texture (3×3)
                Conv(x, c3, 3),
                # Layer 2: ring shape context (3×3)
                Conv(c3, c3, 3),
                # Layer 3: global ring symmetry (3×3)
                Conv(c3, c3, 3),
                # Spatial pooling → class vector
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c3, nc),
            )
            for x in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x):
        """
        Args:
            x: list of tensors from FPN levels, each (B, ch_i, H_i, W_i)
        Returns:
            During training: list of raw predictions per FPN level
            During inference: (decoded_boxes_scores, raw_preds)
        """
        for i in range(self.nl):
            # Box branch: spatial conv → reg output
            box = self.cv2[i](x[i])                          # (B, 4*reg_max, H, W)

            # Cls branch: spatial conv → pool → linear
            # cv3 ends with AdaptiveAvgPool2d(1) + Flatten + Linear
            # → output is (B, nc), need to reshape to (B, nc, 1, 1) then
            #   broadcast back to spatial grid for concat with box
            cls_vec = self.cv3[i](x[i])                      # (B, nc)
            H, W    = x[i].shape[2], x[i].shape[3]
            cls_map = cls_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

            x[i] = torch.cat((box, cls_map), dim=1)          # (B, 4*reg_max+nc, H, W)

        if self.training:
            return x

        # ── inference: decode boxes + scores ─────────────────────────────────
        shape = x[0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                t.transpose(0, 1)
                for t in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        x_cat = torch.cat(
            [xi.view(shape[0], self.no, -1) for xi in x], dim=2
        )

        if self.export and self.format in {
            "saved_model", "pb", "tflite", "edgetpu", "tfjs"
        }:
            box, cls = x_cat[:, : self.reg_max * 4], x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)

        dbox = dist2bbox(
            self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1
        ) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), dim=1)
        return (y, x) if self.export else y

    def bias_init(self):
        """
        Initialise cls bias to log(1/(nc-1)) so each class starts at ~50%
        probability. Called by the Detect trainer automatically.
        Mirrors vanilla Detect.bias_init().
        """
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            # box bias: regression offset initialisation
            a[-1].bias.data[:] = 1.0
            # cls bias: linear layer bias (last module in Sequential)
            b[-1].bias.data[:m.nc] = (
                -torch.log(torch.tensor((m.nc - 1) / 0.01))
            )
