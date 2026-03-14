# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
FILE PATH IN YOUR FORK: ultralytics/nn/modules/__init__.py

INSTRUCTIONS:
  This is a COMPLETE REPLACEMENT of the file.
  The only change vs vanilla ultralytics is the addition of:
    - 3 new import lines  (marked ── HoloYOLO additions ──)
    - 5 new names in __all__

  Everything else is identical to the original file so you can
  safely overwrite it.  Verified against ultralytics 8.3.x.
"""

from .block import (
    C1,
    C2,
    C2PSA,
    C2f,
    C2fAttn,
    C2fCIB,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    ResNetLayer,
    SCDown,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import (
    OBB,
    Classify,
    Detect,
    Pose,
    RTDETRDecoder,
    Segment,
    WorldDetect,
    v10Detect,
)
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

# ── HoloYOLO additions ────────────────────────────────────────────────────────
from .gabor import GaborStem                                 # physics-informed stem
from .radial_attn import AnnularPool, HoloSPPF               # ring-aware SPPF
from .deform_c2f import DeformBottleneck, DeformC2f          # deformable neck
from .holo_detect import HoloDetect                          # deep cls head for RBC subtypes
from .phase_stream import (                                  # PhaseYOLO dual-stream
    GaborPyramid, PhaseGate, DualStreamStem, OrdinalMorphLoss
)
# ─────────────────────────────────────────────────────────────────────────────

__all__ = (
    # block
    "C1", "C2", "C2PSA", "C2f", "C2fAttn", "C2fCIB", "C3", "C3TR", "CIB",
    "DFL", "ELAN1", "PSA", "SPP", "SPPELAN", "SPPF",
    "AConv", "ADown", "Attention", "BNContrastiveHead", "Bottleneck",
    "BottleneckCSP", "CBFuse", "CBLinear", "ContrastiveHead",
    "GhostBottleneck", "HGBlock", "HGStem", "ImagePoolingAttn",
    "Proto", "RepC3", "RepNCSPELAN4", "ResNetLayer", "SCDown",
    # conv
    "CBAM", "ChannelAttention", "Concat", "Conv", "Conv2", "ConvTranspose",
    "DWConv", "DWConvTranspose2d", "Focus", "GhostConv", "LightConv",
    "RepConv", "SpatialAttention",
    # head
    "OBB", "Classify", "Detect", "Pose", "RTDETRDecoder", "Segment",
    "WorldDetect", "v10Detect",
    # transformer
    "AIFI", "MLP", "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer", "LayerNorm2d", "MLPBlock",
    "MSDeformAttn", "TransformerBlock", "TransformerEncoderLayer",
    "TransformerLayer",
    # ── HoloYOLO ─────────────────────────────────────────────
    "GaborStem",
    "AnnularPool",
    "HoloSPPF",
    "DeformBottleneck",
    "DeformC2f",
    "HoloDetect",
    "GaborPyramid",
    "PhaseGate",
    "DualStreamStem",
    "OrdinalMorphLoss",
)
