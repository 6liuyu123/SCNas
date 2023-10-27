import torch
import torch.nn as nn

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation = dilation, groups = C_in, bias = False),
            nn.Conv2d(C_in, C_out, 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm(C_out, affine = affine)
        )
    
    def forward(self, x):
        return self.net(x)
    
class DropPath(nn.Module):
    def __init__(self, p = 0.):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            keep_prob = 1. - self.p
            mask = torch.zeros((x.size(0), 1, 1, 1), device = x.device).beroulli_(keep_prob)
            return x / keep_prob * mask
        return x

"""
    Reduce feature map size by factorized pointwise (stride = 2)
    channels: C_in -> C_out // 2 * 2
    dims: H -> H // 2, W -> W // 2
"""
class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affline = True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride = 2, padding = 0, bias = False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride = 2, padding = 0, bias = False)
        self.bn = nn.BatchNorm2d(C_out, affline = affline)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim = 1)
        out = self.bn(out)
        return out
"""
    AvgPool or MaxPool with BN. 'pool_type' must be 'max' or 'avg'
"""
class PoolBN(nn.Module):
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine = True):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad = False)
        else:
            raise ValueError()
        self.bn = nn.BatchNorm2d(C, affine = affine)
    
    def forward(self, x):
        out = self.pool(x)

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine = True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation = 1, affine = affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation = 1, affine = affine),
        )
    
    def forward(self, x):
        return self.net(x)

"""
    Standard conv: ReLU - Conv - BN
"""
class StdConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affline = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(C_out, affline = affline)
        )

    def forward(self, x):
        return self.net(x)