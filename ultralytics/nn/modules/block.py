# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "SCDPatchEmbed", "SCDPatchMerge", "SCDSwinStage", "SCDAspp", "SCDBiFPN",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
)


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize RepC3 module with RepConv blocks.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and addition to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 4 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes image features through unchanged after fusing."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with three convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth-wise convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the fused RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        if not hasattr(self, "conv1"):
            return  # already fused
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Compact Inverted Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use large kernel. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use large kernel.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c2.
        m (nn.ModuleList): List of PSABlock modules for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided into.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.all_head_dim = all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, all_head_dim, 7, 1, 3, g=all_head_dim, act=False)

    def __setstate__(self, state):
        """Add missing all_head_dim attribute to old checkpoints."""
        super().__setstate__(state)
        if not hasattr(self, "all_head_dim"):
            self.all_head_dim = self.head_dim * self.num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, _, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, self.all_head_dim * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, self.all_head_dim)
            v = v.reshape(B // self.area, N * self.area, self.all_head_dim)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided into.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class Proto26(Proto):
    """Ultralytics YOLO26 models mask Proto module for segmentation models."""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
            nc (int): Number of classes for semantic segmentation.
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))

    def forward(self, x: torch.Tensor, return_semantic: bool = True) -> torch.Tensor:
        """Perform a forward pass by fusing multi-scale feature maps and generating proto masks."""
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = f(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        p = super().forward(self.feat_fuse(feat))
        if self.training and return_semantic:
            semantic = self.semseg(feat)
            return (p, semantic)
        return p

    def fuse(self):
        """Fuse the model for inference by removing the semantic segmentation head."""
        self.semseg = None


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model.

    References:
        https://arxiv.org/abs/1605.08803
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
    """

    @staticmethod
    def nets():
        """Get the scale model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """Get the translation model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return torch.distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super().__init__()

        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping from the data space to the latent space and calculate the log determinant of the Jacobian
        matrix.
        """
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""
        if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
            self.float()
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det





"""
DHM custom blocks for ultralytics/nn/modules/block.py
======================================================
INSTRUCTION: Copy everything below the dashed line and APPEND it to the
bottom of  ultralytics/nn/modules/block.py  in your repo.

Blocks defined here:
  - RSFiLMGenerator        : Rayleigh-Sommerfeld physics -> (gamma, beta) per stage
  - PatchEmbedDHM          : Conv2d patch embedding (replaces stem convolutions)
  - PatchMergingDHM        : 2x spatial downsampling via pixel-shuffle concat
  - WindowAttentionDHM     : Relative-position-bias window attention
  - SwinBlockDHM           : W-MSA + SW-MSA pair with FiLM injection
  - MWSwinStageDHM         : One backbone stage: fixed window_size, depth blocks
  - BiFPNLayerDHM          : One bidirectional weighted-fusion PANet layer
  - ASPPModuleDHM          : Atrous Spatial Pyramid Pooling
  - LapPyramidDecoderDHM   : Laplacian sub-band residual decoder
"""

# ── standard imports (already present in block.py, listed for clarity) ──────
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ════════════════════════════════════════════════════════════════════════════
#  RS-FiLM PHYSICS ENCODER
# ════════════════════════════════════════════════════════════════════════════

def _rs_physics_features(wavelength, L_value, z_value,
                          pixel_size: float = 3.8e-3,
                          N_pix: int = 768):
    """
    Compute 8-dim Rayleigh-Sommerfeld physics features.

    Parameters (all tensors on same device as model)
    ----------
    wavelength : (B,)  μm
    L_value    : (B,)  mm  — camera-to-source distance
    z_value    : (B,)  mm  — sample-to-source distance
    pixel_size : float μm  — detector pixel pitch (default 3.8 μm)
    N_pix      : int       — image side length in pixels (default 768)

    Returns
    -------
    feat : (B, 8)  float32
    """
    lam    = wavelength                         # μm
    z      = z_value  * 1e3                    # mm → μm
    L      = L_value  * 1e3                    # mm → μm
    px     = pixel_size                         # μm  scalar
    k      = 2.0 * math.pi / lam              # (B,) wavenumber 1/μm

    # Half-width of hologram aperture
    W_half = (N_pix * px) * 0.5               # scalar μm  (=1459.2 μm)

    # Maximum off-axis distance (corner pixel)
    r_max  = torch.sqrt(
        torch.full_like(z, W_half ** 2) + z ** 2
    )                                          # (B,) μm

    # 1. RS obliquity factor: cos(θ_max) = z / r_max  ∈ (0,1]
    obliquity  = z / r_max

    # 2. Phase at corner: k * r_max  (radians, large)
    phi_corner = k * r_max

    # 3. On-axis phase: k * z
    phi_onaxis = k * z

    # 4. Near/far-field ratio: z / λ  (log-compressed)
    z_over_lam = z / lam

    # 5. Magnification: M = L / z
    mag = L / z.clamp(min=1e-3)

    # 6. Object-plane effective pixel: px / M  (μm)
    px_obj = torch.full_like(lam, px) / mag.clamp(min=1e-3)

    # 7. Phase wrap count: φ_corner / 2π
    wrap_count = phi_corner / (2.0 * math.pi)

    # 8. Phase adequacy: φ_corner / π
    phase_adequacy = phi_corner / math.pi

    feat = torch.stack([
        obliquity,                                   # ∈ (0,1]
        torch.log1p(phi_corner.clamp(min=0)),        # ≥ 0
        torch.log1p(phi_onaxis.clamp(min=0)),        # ≥ 0
        torch.log1p(z_over_lam),                     # ≥ 0
        torch.log1p(mag.clamp(max=1e4)),             # ≥ 0
        torch.log1p(px_obj.clamp(max=1e4)),          # ≥ 0
        torch.sin(phi_onaxis % (2 * math.pi)),       # cyclic cue
        torch.cos(phi_onaxis % (2 * math.pi)),       # cyclic cue
    ], dim=-1)                                       # (B, 8)
    return feat


class RSFiLMGenerator(nn.Module):
    """
    Maps RS physics features → per-stage (gamma, beta) for FiLM modulation.

    Args
    ----
    physics_dim  : int   input feature dimension (8 RS features)
    hidden_dim   : int   shared MLP hidden size
    stage_dims   : list  channel count at each backbone stage to be modulated
    pixel_size   : float detector pixel pitch in μm
    n_pix        : int   image side in pixels
    """
    def __init__(self, physics_dim: int = 8, hidden_dim: int = 128,
                 stage_dims: list = None,
                 pixel_size: float = 3.8e-3, n_pix: int = 768):
        super().__init__()
        self.pixel_size = pixel_size
        self.n_pix      = n_pix

        self.shared = nn.Sequential(
            nn.Linear(physics_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # separate gamma/beta head per stage
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 2 * d) for d in stage_dims
        ])

    def forward(self, wavelength, L_value, z_value):
        """
        Returns list of (gamma, beta) tensors, one per stage.
        gamma, beta: (B, stage_dim)
        """
        feat = _rs_physics_features(
            wavelength, L_value, z_value,
            self.pixel_size, self.n_pix
        )                                    # (B, 8)
        h = self.shared(feat)                # (B, hidden)
        out = []
        for head in self.heads:
            gb = head(h)
            gamma, beta = gb.chunk(2, dim=-1)
            out.append((gamma, beta))
        return out                           # list[(B,C), (B,C)] × n_stages


# ════════════════════════════════════════════════════════════════════════════
#  PATCH EMBEDDING
# ════════════════════════════════════════════════════════════════════════════

class PatchEmbedDHM(nn.Module):
    """
    Conv2d-based patch embedding: image → token sequence.

    Args
    ----
    in_chans    : int  input channels (1 for grayscale hologram)
    embed_dim   : int  token dimension C
    patch_size  : int  downsampling factor (default 4 → 768→192)
    """
    def __init__(self, in_chans: int = 1, embed_dim: int = 96,
                 patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C_in, H, W)
        x = self.proj(x)                           # (B, C, H/ps, W/ps)
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)           # (B, Hp*Wp, C)
        x = self.norm(x)
        return x, Hp, Wp


# ════════════════════════════════════════════════════════════════════════════
#  PATCH MERGING  (2× downsampling inside the backbone)
# ════════════════════════════════════════════════════════════════════════════

class PatchMergingDHM(nn.Module):
    """Pixel-shuffle concatenation → linear reduction: (B,H*W,C) → (B,H/2*W/2,2C)."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x  = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x  = x.view(B, -1, 4 * C)
        x  = self.norm(x)
        x  = self.reduction(x)                     # (B, H/2*W/2, 2C)
        return x, H // 2, W // 2


# ════════════════════════════════════════════════════════════════════════════
#  WINDOW ATTENTION
# ════════════════════════════════════════════════════════════════════════════

def _window_partition(x, ws):
    """(B,H,W,C) → (B*nW, ws, ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0,1,3,2,4,5).contiguous().view(-1, ws, ws, C)


def _window_reverse(windows, ws, H, W):
    """(B*nW, ws, ws, C) → (B, H, W, C)"""
    B = int(windows.shape[0] / (H // ws * W // ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)


class WindowAttentionDHM(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.ws        = window_size
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5

        # relative position bias table
        self.rpb = nn.Parameter(
            torch.zeros((2*window_size-1)**2, num_heads)
        )
        nn.init.trunc_normal_(self.rpb, std=0.02)

        coords   = torch.arange(window_size)
        grid     = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        flat     = torch.flatten(grid, 1)
        rel      = flat[:, :, None] - flat[:, None, :]
        rel      = rel.permute(1, 2, 0).contiguous()
        rel[:,:,0] += window_size - 1
        rel[:,:,1] += window_size - 1
        rel[:,:,0] *= 2 * window_size - 1
        self.register_buffer('rpi', rel.sum(-1))   # relative position index

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2,-1)

        bias = self.rpb[self.rpi.view(-1)].view(
            self.ws**2, self.ws**2, -1).permute(2,0,1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x    = (attn @ v).transpose(1,2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ════════════════════════════════════════════════════════════════════════════
#  SWIN BLOCK  (W-MSA + SW-MSA pair, FiLM injected after MLP)
# ════════════════════════════════════════════════════════════════════════════

class _DropPath(nn.Module):
    def __init__(self, p=0.):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0. or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand  = torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep * torch.floor(rand + keep)


class SwinBlockDHM(nn.Module):
    """
    Single Swin Transformer block.
    FiLM (gamma, beta) is applied after the second LayerNorm+MLP step.
    """
    def __init__(self, dim, num_heads, window_size, shift_size=0,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim        = dim
        self.ws         = window_size
        self.shift      = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttentionDHM(dim, window_size, num_heads,
                                        attn_drop=attn_drop, proj_drop=drop)
        self.dp    = _DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hid        = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, dim), nn.Dropout(drop),
        )
        self.H = self.W = None   # set by parent stage before forward

    def _attn_mask(self, H, W, device):
        if self.shift == 0:
            return None
        img = torch.zeros(1, H, W, 1, device=device)
        hs  = (slice(0,-self.ws), slice(-self.ws,-self.shift), slice(-self.shift,None))
        ws  = (slice(0,-self.ws), slice(-self.ws,-self.shift), slice(-self.shift,None))
        cnt = 0
        for h in hs:
            for w in ws:
                img[:,h,w,:] = cnt; cnt += 1
        wins = _window_partition(img, self.ws).view(-1, self.ws**2)
        mask = wins.unsqueeze(1) - wins.unsqueeze(2)
        return mask.masked_fill(mask!=0, -100.).masked_fill(mask==0, 0.)

    def forward(self, x, film_gamma=None, film_beta=None):
        H, W   = self.H, self.W
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift > 0:
            x = torch.roll(x, (-self.shift,-self.shift), (1,2))

        wins   = _window_partition(x, self.ws).view(-1, self.ws**2, C)
        mask   = self._attn_mask(H, W, x.device)
        wins   = self.attn(wins, mask=mask)
        x      = _window_reverse(wins.view(-1, self.ws, self.ws, C), self.ws, H, W)

        if self.shift > 0:
            x = torch.roll(x, (self.shift, self.shift), (1,2))

        x = shortcut + self.dp(x.view(B, L, C))
        x = x + self.dp(self.mlp(self.norm2(x)))

        # FiLM modulation after MLP
        if film_gamma is not None and film_beta is not None:
            x = film_gamma.unsqueeze(1) * x + film_beta.unsqueeze(1)

        return x


# ════════════════════════════════════════════════════════════════════════════
#  MULTI-WINDOW SWIN STAGE
# ════════════════════════════════════════════════════════════════════════════

class MWSwinStageDHM(nn.Module):
    """
    One backbone stage with a fixed window_size and depth W/SW block pairs.

    Args
    ----
    dim         : int   token channels
    num_heads   : int   attention heads
    window_size : int   window size (4, 8, or 12 in your config)
    depth       : int   number of Swin blocks (must be even: pairs of W+SW)
    downsample  : bool  apply PatchMerging at the end
    film_dim    : int   expected channel dim for FiLM (same as dim before merge)
    drop_path   : list  stochastic depth rates, length == depth
    """
    def __init__(self, dim, num_heads, window_size, depth=2,
                 downsample=True, mlp_ratio=4.,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.ws    = window_size
        self.shift = window_size // 2
        self.depth = depth

        dp = drop_path if isinstance(drop_path, list) else [drop_path] * depth
        self.blocks = nn.ModuleList([
            SwinBlockDHM(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if i % 2 == 0 else self.shift,
                mlp_ratio=mlp_ratio, drop=drop,
                attn_drop=attn_drop, drop_path=dp[i],
            )
            for i in range(depth)
        ])
        self.merge = PatchMergingDHM(dim) if downsample else None

    def forward(self, x, H, W, film_gamma=None, film_beta=None):
        """
        x            : (B, H*W, C)
        film_gamma   : (B, C) — applied after last block in this stage
        film_beta    : (B, C)
        Returns      : x_out (B,H*W,C),  H,  W,
                       x_down (B,H/2*W/2,2C),  Hd,  Wd
        """
        for i, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W
            # apply FiLM only on last block of stage
            last = (i == len(self.blocks) - 1)
            x = blk(x,
                    film_gamma if last else None,
                    film_beta  if last else None)

        if self.merge is not None:
            x_down, Hd, Wd = self.merge(x, H, W)
            return x, H, W, x_down, Hd, Wd

        return x, H, W, x, H, W   # no downsample → pass through


# ════════════════════════════════════════════════════════════════════════════
#  BiFPN LAYER
# ════════════════════════════════════════════════════════════════════════════

class _BiFPNConv(nn.Module):
    """Depthwise-separable conv used inside BiFPN nodes."""
    def __init__(self, c):
        super().__init__()
        self.dw  = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.pw  = nn.Conv2d(c, c, 1, bias=False)
        self.bn  = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class BiFPNLayerDHM(nn.Module):
    """
    One complete BiFPN layer over three feature levels P3/P4/P5.

    All three input maps are projected to `neck_dim` channels first.
    Weighted fusion uses fast normalised attention (ε-softmax).

    Args
    ----
    in_dims  : tuple(int,int,int)  input channel counts (P3, P4, P5)
    neck_dim : int                  unified channel count inside BiFPN
    """
    def __init__(self, in_dims=(96, 192, 384), neck_dim=256):
        super().__init__()
        d0, d1, d2 = in_dims
        eps = 1e-4

        # lateral projections (only used on first call; reused if same dims)
        self.lat0 = nn.Sequential(
            nn.Conv2d(d0, neck_dim, 1, bias=False),
            nn.BatchNorm2d(neck_dim), nn.SiLU(inplace=True))
        self.lat1 = nn.Sequential(
            nn.Conv2d(d1, neck_dim, 1, bias=False),
            nn.BatchNorm2d(neck_dim), nn.SiLU(inplace=True))
        self.lat2 = nn.Sequential(
            nn.Conv2d(d2, neck_dim, 1, bias=False),
            nn.BatchNorm2d(neck_dim), nn.SiLU(inplace=True))

        # top-down (td) fusion weights: (w_in, w_skip)
        self.w_td1 = nn.Parameter(torch.ones(2))  # P4_td  = f(P5_up, P4)
        self.w_td0 = nn.Parameter(torch.ones(2))  # P3_td  = f(P4_td_up, P3)
        # bottom-up (bu) fusion weights
        self.w_bu1 = nn.Parameter(torch.ones(3))  # P4_out = f(P4_td, P3_bu_down, P4)
        self.w_bu2 = nn.Parameter(torch.ones(2))  # P5_out = f(P5, P4_out_down)

        self.td1_conv  = _BiFPNConv(neck_dim)
        self.td0_conv  = _BiFPNConv(neck_dim)
        self.bu1_conv  = _BiFPNConv(neck_dim)
        self.bu2_conv  = _BiFPNConv(neck_dim)

        self.eps = eps
        self._lateral_done = False

    def _w(self, raw):
        """Fast normalised weights (all positive, sum to 1)."""
        w = F.relu(raw)
        return w / (w.sum() + self.eps)

    def forward(self, features):
        """
        features : [P3, P4, P5]
        Returns  : [P3_out, P4_out, P5_out]
        """
    
        p0, p1, p2 = features
    
        # ------------------------------------------------------------------
        # LATERAL PROJECTION
        # ------------------------------------------------------------------
        if p0.shape[1] != self.lat0[0].out_channels:
            p0 = self.lat0(p0)
            p1 = self.lat1(p1)
            p2 = self.lat2(p2)
    
        # Debug shapes
        # print("P3:", p0.shape)
        # print("P4:", p1.shape)
        # print("P5:", p2.shape)
    
        # ------------------------------------------------------------------
        # TOP-DOWN PATH
        # ------------------------------------------------------------------
    
        # P5 → P4
        w = self._w(self.w_td1)
    
        p5_up = F.interpolate(
            p2,
            size=p1.shape[-2:],
            mode='nearest'
        )
    
        p4_td = self.td1_conv(
            w[0] * p1 +
            w[1] * p5_up
        )
    
        # P4 → P3
        w = self._w(self.w_td0)
    
        p4_up = F.interpolate(
            p4_td,
            size=p0.shape[-2:],
            mode='nearest'
        )
    
        p3_td = self.td0_conv(
            w[0] * p0 +
            w[1] * p4_up
        )
    
        # ------------------------------------------------------------------
        # BOTTOM-UP PATH
        # ------------------------------------------------------------------
    
        # P3 → P4
        w = self._w(self.w_bu1)
    
        p3_down = F.avg_pool2d(
            p3_td,
            kernel_size=2,
            stride=2
        )
    
        p4_out = self.bu1_conv(
            w[0] * p1 +
            w[1] * p4_td +
            w[2] * p3_down
        )
    
        # P4 → P5
        w = self._w(self.w_bu2)
    
        p4_down = F.avg_pool2d(
            p4_out,
            kernel_size=2,
            stride=2
        )
    
        p5_out = self.bu2_conv(
            w[0] * p2 +
            w[1] * p4_down
        )
    
        return [p3_td, p4_out, p5_out]

# ════════════════════════════════════════════════════════════════════════════
#  ASPP MODULE
# ════════════════════════════════════════════════════════════════════════════

class ASPPModuleDHM(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    Dilation rates chosen to match typical Fresnel zone ring radii at
    λ=0.405 μm, z=0.36–1.17 mm, px=3.8 μm.

    Args
    ----
    in_channels  : int
    out_channels : int
    dilations    : tuple  default (1, 2, 4, 8) — adapt to your fringe scale
    """
    def __init__(self, in_channels: int, out_channels: int,
                 dilations: tuple = (1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList()

        # 1×1 conv (dilation=1)
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
        ))
        # dilated 3×3 convs
        for d in dilations[1:]:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
            ))
        # global average pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
        )
        total_branches = len(dilations) + 1   # dilated + GAP
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * total_branches, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        h, w   = x.shape[-2:]
        feats  = [b(x) for b in self.branches]
        gap_up = F.interpolate(self.gap(x), size=(h, w),
                               mode='bilinear', align_corners=False)
        feats.append(gap_up)
        return self.project(torch.cat(feats, dim=1))


# ════════════════════════════════════════════════════════════════════════════
#  LAPLACIAN PYRAMID DECODER
# ════════════════════════════════════════════════════════════════════════════

class LapPyramidDecoderDHM(nn.Module):
    """
    Laplacian sub-band residual decoder.

    Reconstructs phase progressively:
      level 3 (P5, 48²) → base estimate
      level 2 (P4, 96²) → add sub-band residual
      level 1 (P3, 192²) → add sub-band residual
      upsample all → 768² final phase map

    Each level computes:
        x_up   = bilinear_upsample(prev)
        res    = conv(current_feature - x_up)   ← Laplacian residual
        out    = x_up + res

    Args
    ----
    neck_dim     : int   channel count from BiFPN (all 3 levels same)
    mid_channels : int   intermediate channels in residual convs
    """
    def __init__(self, neck_dim: int = 256, mid_channels: int = 128):
        super().__init__()
        # base at P5 (coarsest)
        self.base_conv = nn.Sequential(
            nn.Conv2d(neck_dim, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),  # → single-channel phase estimate
        )
        # residual at P4
        self.res2_conv = nn.Sequential(
            nn.Conv2d(neck_dim + 1, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
        )
        # residual at P3
        self.res1_conv = nn.Sequential(
            nn.Conv2d(neck_dim + 1, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
        )
        # final refinement to 768×768
        self.final_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, neck_feats, target_size=(768, 768)):
        """
        neck_feats : [P3(192²), P4(96²), P5(48²)] all (B, neck_dim, H, W)
        Returns    : (B, 1, 768, 768) phase map
        """
        p3, p4, p5 = neck_feats

        # Level 3: base estimate at 48×48
        base   = self.base_conv(p5)                          # (B,1,48,48)

        # Level 2: upsample base → 96², compute Laplacian residual with P4
        up2    = F.interpolate(base, size=p4.shape[-2:],
                               mode='bilinear', align_corners=False)
        lap2   = self.res2_conv(torch.cat([p4, up2], dim=1)) # (B,1,96,96)
        out2   = up2 + lap2

        # Level 1: upsample out2 → 192², compute Laplacian residual with P3
        up1    = F.interpolate(out2, size=p3.shape[-2:],
                               mode='bilinear', align_corners=False)
        lap1   = self.res1_conv(torch.cat([p3, up1], dim=1)) # (B,1,192,192)
        out1   = up1 + lap1

        # Final: upsample to 768² and refine
        out    = F.interpolate(out1, size=target_size,
                               mode='bilinear', align_corners=False)
        return self.final_conv(out)                          # (B,1,768,768)




# ============================================================
# SCD (Sickle Cell Detection) custom blocks
# Replace/append to ultralytics/nn/modules/block.py
# ============================================================


def _scd_window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) → (B*nW, ws*ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)


def _scd_window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, ws*ws, C) → (B, H, W, C)"""
    C = windows.shape[-1]
    B = int(windows.shape[0] / (H // ws * W // ws))
    return (windows.view(B, H // ws, W // ws, ws, ws, C)
            .permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C))


class _SCDDropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(
            torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


# ─────────────────────────────────────────────────────────────
# SCDPatchEmbed
# ─────────────────────────────────────────────────────────────

class SCDPatchEmbed(nn.Module):
    """
    Convolutional patch embedding.
    (B, in_chans, H, W) → (B, embed_dim, H/patch_size, W/patch_size)

    parse_model args injection: YAML args=[embed_dim, patch_size],
    parse_model prepends c1 → __init__(in_chans, embed_dim, patch_size).
    """

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size,
                              bias=False)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


# ─────────────────────────────────────────────────────────────
# SCDPatchMerge
# ─────────────────────────────────────────────────────────────

class SCDPatchMerge(nn.Module):
    """
    2× spatial downsampling: (B, C, H, W) → (B, 2C, H/2, W/2).
    Uses pixel-shuffle concat + LayerNorm + Linear projection.

    parse_model args injection: YAML args=[], parse_model passes [ch[f]] → __init__(dim).
    Output channels c2 = 2 * dim (declared in tasks.py parse_model override).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.proj = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Pad to even spatial dims if needed
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, H, W = x.shape
        t = x.permute(0, 2, 3, 1)                          # (B, H, W, C)
        t = torch.cat([t[:, 0::2, 0::2],
                       t[:, 1::2, 0::2],
                       t[:, 0::2, 1::2],
                       t[:, 1::2, 1::2]], dim=-1)           # (B, H/2, W/2, 4C)
        t = self.proj(self.norm(t))                         # (B, H/2, W/2, 2C)
        return t.permute(0, 3, 1, 2).contiguous()


# ─────────────────────────────────────────────────────────────
# Window Attention
# ─────────────────────────────────────────────────────────────

class _SCDWindowAttn(nn.Module):
    """Window MHSA with relative positional bias. Table is built for ws×ws."""

    def __init__(self, dim: int, ws: int, num_heads: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.ws = ws
        self.nh = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.rpb = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rpb, std=0.02)

        coords = torch.arange(ws)
        gy, gx = torch.meshgrid(coords, coords, indexing='ij')
        flat = torch.stack([gy.flatten(), gx.flatten()])    # (2, ws^2)
        rel = flat[:, :, None] - flat[:, None, :]           # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[..., 0] += ws - 1
        rel[..., 1] += ws - 1
        rel[..., 0] *= 2 * ws - 1
        self.register_buffer('rpi', rel.sum(-1))            # (ws^2, ws^2)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
    
        B_, N, C = x.shape
    
        nh = self.nh
        hd = C // nh
    
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, nh, hd)
            .permute(2, 0, 3, 1, 4)
        )
    
        q, k, v = qkv.unbind(0)
    
        # --------------------------------------------------
        # Attention logits
        # --------------------------------------------------
    
        attn = (q * self.scale) @ k.transpose(-2, -1)
    
        # --------------------------------------------------
        # Relative position bias
        # --------------------------------------------------
    
        bias = (
            self.rpb[self.rpi.view(-1)]
            .view(self.ws * self.ws,
                  self.ws * self.ws,
                  nh)
            .permute(2, 0, 1)
            .contiguous()
        )
    
        # IMPORTANT: match AMP dtype
        bias = bias.to(attn.dtype)
    
        attn = attn + bias.unsqueeze(0)
    
        # --------------------------------------------------
        # Shifted-window mask
        # --------------------------------------------------
    
        if mask is not None:
    
            mask = mask.to(attn.dtype)
    
            nW = mask.shape[0]
    
            attn = attn.view(
                B_ // nW,
                nW,
                nh,
                N,
                N
            )
    
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
    
            attn = attn.view(
                -1,
                nh,
                N,
                N
            )
    
        # --------------------------------------------------
        # Softmax
        # --------------------------------------------------
    
        attn = self.attn_drop(
            attn.softmax(dim=-1)
        )
    
        # IMPORTANT: ensure same dtype
        v = v.to(attn.dtype)
    
        # --------------------------------------------------
        # Attention output
        # --------------------------------------------------
    
        out = (
            attn @ v
        ).transpose(
            1, 2
        ).reshape(
            B_,
            N,
            C
        )
    
        out = self.proj(out)
    
        out = self.proj_drop(out)
    
        return out


# ─────────────────────────────────────────────────────────────
# Swin Block — with dynamic padding/unpadding
# ─────────────────────────────────────────────────────────────

class _SCDSwinBlock(nn.Module):
    """
    Single Swin Transformer block operating on (B, C, H, W).

    Pads H and W to the nearest multiple of window_size before partitioning,
    then removes the padding from the output. This guarantees correctness
    regardless of whether the spatial dimensions are exact multiples of ws.
    """

    def __init__(self, dim: int, num_heads: int, ws: int,
                 shift: int = 0, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.ws = ws
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SCDWindowAttn(dim, ws, num_heads,
                                   attn_drop=attn_drop, proj_drop=drop)
        self.dp = _SCDDropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hid, dim), nn.Dropout(drop),
        )

    def _pad_to_window(self, x: torch.Tensor) -> tuple:
        """Pad (B,H,W,C) so H and W are multiples of ws. Returns (padded, H_orig, W_orig)."""
        H, W = x.shape[1], x.shape[2]
        pad_h = (self.ws - H % self.ws) % self.ws
        pad_w = (self.ws - W % self.ws) % self.ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))   # pad (C=last dim untouched)
        return x, H, W

    def _mask(self, Hp: int, Wp: int,
              device: torch.device) -> torch.Tensor | None:
        """Build cyclic-shift attention mask on padded spatial dims Hp×Wp."""
        if self.shift == 0:
            return None
        ws, s = self.ws, self.shift
        img = torch.zeros(1, Hp, Wp, 1, device=device)
        cnt = 0
        for hs in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
            for ws_ in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                img[:, hs, ws_, :] = cnt
                cnt += 1
        wins = _scd_window_partition(img, ws).squeeze(-1)   # (nW, ws^2)
        m = wins.unsqueeze(1) - wins.unsqueeze(2)
        return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        skip = x

        # Convert to (B, H, W, C) token layout
        t = x.permute(0, 2, 3, 1)          # (B, H, W, C)

        # Pad to window-size multiple
        t, H_orig, W_orig = self._pad_to_window(t)
        Hp, Wp = t.shape[1], t.shape[2]

        t = self.norm1(t)

        # Cyclic shift on padded map
        if self.shift:
            t = torch.roll(t, shifts=(-self.shift, -self.shift), dims=(1, 2))

        # Partition → attend → reverse
        wins = _scd_window_partition(t, self.ws)
        wins = self.attn(wins, mask=self._mask(Hp, Wp, x.device))
        t = _scd_window_reverse(wins, self.ws, Hp, Wp)     # (B, Hp, Wp, C)

        # Reverse cyclic shift
        if self.shift:
            t = torch.roll(t, shifts=(self.shift, self.shift), dims=(1, 2))

        # Remove padding
        t = t[:, :H_orig, :W_orig, :].contiguous()         # (B, H, W, C)

        # Back to (B, C, H, W) and residual
        t = t.permute(0, 3, 1, 2).contiguous()
        x = skip + self.dp(t)

        # MLP branch
        t = self.norm2(x.permute(0, 2, 3, 1))
        t = self.mlp(t).permute(0, 3, 1, 2).contiguous()
        return x + self.dp(t)


# ─────────────────────────────────────────────────────────────
# SCDSwinStage  (YAML-registered module)
# ─────────────────────────────────────────────────────────────

class SCDSwinStage(nn.Module):
    """
    Multi-window Swin Transformer stage on 4-D (B, C, H, W) feature maps.

    Stacks `depth` W/SW-MSA block pairs. Output channels == input channels.
    Spatial downsampling is done by a separate SCDPatchMerge layer.

    Each block pads its input to the nearest ws multiple and unpads the output,
    so this stage works on any spatial size ≥ ws.

    parse_model injection: YAML args=[window_size, depth, ...],
    tasks.py prepends dim=c1 → __init__(dim, window_size, depth, ...).

    Args:
        dim        : channel count (== c1 from parse_model)
        num_heads  : derived as max(dim // 32, 1) — no YAML arg needed
        window_size: ws (must be ≤ spatial size; padding handles non-multiples)
        depth      : number of blocks
        mlp_ratio  : MLP expansion ratio
        drop_path  : peak stochastic-depth rate
    """

    def __init__(self, dim: int, window_size: int = 8, depth: int = 2,
                 mlp_ratio: float = 4.0, drop: float = 0.0,
                 attn_drop: float = 0.0, drop_path: float = 0.1):
        super().__init__()
        num_heads = max(dim // 32, 1)
        shift = window_size // 2
        dp_rates = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            _SCDSwinBlock(
                dim=dim, num_heads=num_heads, ws=window_size,
                shift=0 if i % 2 == 0 else shift,
                mlp_ratio=mlp_ratio, drop=drop,
                attn_drop=attn_drop, drop_path=dp_rates[i],
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# ─────────────────────────────────────────────────────────────
# SCDAspp  (YAML-registered module)
# ─────────────────────────────────────────────────────────────

class SCDAspp(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    (B, in_channels, H, W) → (B, out_channels, H, W)

    parse_model injection: YAML args=[out_channels],
    tasks.py prepends in_channels=c1 → __init__(in_channels, out_channels, ...).

    Args:
        in_channels : c1, injected by parse_model
        out_channels: YAML args[0]
        dilations   : dilation rates (default (1, 2, 4, 8))
    """

    def __init__(self, in_channels: int, out_channels: int,
                 dilations: tuple = (1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
            )
        ])
        for d in dilations[1:]:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
            ))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
        )
        n = len(dilations) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * n, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        feats.append(F.interpolate(self.gap(x), size=(h, w),
                                   mode='bilinear', align_corners=False))
        return self.project(torch.cat(feats, dim=1))


# ─────────────────────────────────────────────────────────────
# SCDBiFPN  (YAML-registered module)
# ─────────────────────────────────────────────────────────────

class _SCDBiFPNNode(nn.Module):
    """Depthwise-separable BN+SiLU fusion node."""

    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class SCDBiFPN(nn.Module):
    """
    One BiFPN layer over three feature levels [P3, P4, P5].

    Accepts and returns a Python list of three 4-D tensors.
    Lateral projections align all levels to neck_dim channels on first call.
    Fast normalised fusion weights (ε-softmax) are used.

    parse_model injection: tasks.py passes in_dims=tuple(ch[x] for x in f)
    as first arg → __init__(in_dims, neck_dim).

    Args:
        in_dims : (P3_ch, P4_ch, P5_ch), injected by tasks.py
        neck_dim: unified channel count (YAML args[0])
    """

    _EPS = 1e-4

    def __init__(self, in_dims: tuple, neck_dim: int = 192):
        super().__init__()
        self.nd = neck_dim

        self.lat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, neck_dim, 1, bias=False),
                nn.BatchNorm2d(neck_dim), nn.SiLU(inplace=True),
            ) if d != neck_dim else nn.Identity()
            for d in in_dims
        ])

        self.w_td4 = nn.Parameter(torch.ones(2))
        self.w_td3 = nn.Parameter(torch.ones(2))
        self.w_bu4 = nn.Parameter(torch.ones(3))
        self.w_bu5 = nn.Parameter(torch.ones(2))

        self.node_td4 = _SCDBiFPNNode(neck_dim)
        self.node_td3 = _SCDBiFPNNode(neck_dim)
        self.node_bu4 = _SCDBiFPNNode(neck_dim)
        self.node_bu5 = _SCDBiFPNNode(neck_dim)

    def _w(self, raw: nn.Parameter) -> torch.Tensor:
        w = F.relu(raw)
        return w / (w.sum() + self._EPS)

    def forward(self, features: list) -> list:
        p3, p4, p5 = [self.lat[i](f) for i, f in enumerate(features)]

        # Top-down
        w = self._w(self.w_td4)
        p4_td = self.node_td4(
            w[0] * p4 + w[1] * F.interpolate(p5, size=p4.shape[-2:], mode='nearest'))

        w = self._w(self.w_td3)
        p3_td = self.node_td3(
            w[0] * p3 + w[1] * F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest'))

        # Bottom-up
        w = self._w(self.w_bu4)
        p4_out = self.node_bu4(
            w[0] * p4 + w[1] * p4_td +
            w[2] * F.avg_pool2d(p3_td, kernel_size=2, stride=2))

        w = self._w(self.w_bu5)
        p5_out = self.node_bu5(
            w[0] * p5 + w[1] * F.avg_pool2d(p4_out, kernel_size=2, stride=2))

        return [p3_td, p4_out, p5_out]
