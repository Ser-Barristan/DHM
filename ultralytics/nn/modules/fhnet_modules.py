import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import DWConv


# ------------------------------------------------
# Lightweight C2f using existing DWConv
# ------------------------------------------------



import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import DWConv

class DSC2f(nn.Module):

    def __init__(self, c1, c2, *args, **kwargs):
        super().__init__()

        # default number of internal layers
        n = 1
        if len(args) > 0:
            try:
                n = int(args[0])
            except:
                n = 1

        self.cv1 = DWConv(c1, c2, 3, 1)

        self.m = nn.ModuleList(
            DWConv(c2, c2, 3, 1) for _ in range(n)
        )

        self.cv2 = DWConv(c2*(n+1), c2, 1, 1)

    def forward(self, x):

        y = [self.cv1(x)]

        for m in self.m:
            y.append(m(y[-1]))

        return self.cv2(torch.cat(y,1))
# ------------------------------------------------
# SimAM attention (parameter-free)
# ------------------------------------------------

import torch
import torch.nn as nn

class SimAM(nn.Module):

    def __init__(self, c1=None, c2=None, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):

        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu = x - x.mean(dim=[2,3], keepdim=True)

        var = (x_minus_mu ** 2).sum(dim=[2,3], keepdim=True) / n

        y = x_minus_mu / torch.sqrt(var + self.e_lambda)

        return x * torch.sigmoid(y)

# ------------------------------------------------
# Fringe Feature Block (for holographic fringes)
# ------------------------------------------------

class FringeBlock(nn.Module):

    def __init__(self, c):
        super().__init__()

        laplace = torch.tensor(
            [[0,1,0],
             [1,-4,1],
             [0,1,0]], dtype=torch.float32
        )

        self.register_buffer(
            "lap",
            laplace.view(1,1,3,3).repeat(c,1,1,1)
        )

        self.conv = nn.Conv2d(c*2, c, 1)

    def forward(self, x):

        edge = F.conv2d(x, self.lap, padding=1, groups=x.shape[1])

        out = torch.cat([x, edge],1)

        return self.conv(out)
