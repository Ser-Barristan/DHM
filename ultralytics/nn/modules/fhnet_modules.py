import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import DWConv


# ------------------------------------------------
# Lightweight C2f using existing DWConv
# ------------------------------------------------

class DSC2f(nn.Module):

    def __init__(self, c1, c2, n=1):
        super().__init__()

        self.cv1 = DWConv(c1, c2, 3, 1)

        self.m = nn.ModuleList(
            [DWConv(c2, c2, 3, 1) for _ in range(n)]
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

class SimAM(nn.Module):

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):

        b, c, h, w = x.size()

        n = w*h - 1

        x_mu = x.mean(dim=[2,3], keepdim=True)

        var = ((x-x_mu)**2).sum(dim=[2,3], keepdim=True) / n

        y = (x-x_mu) / torch.sqrt(var + self.e_lambda)

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
