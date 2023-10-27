from AuxiliaryHead import AuxiliaryHead
from Cell import Cell
from Ops import DropPath

import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size, in_channels, channels, n_classes, n_layers, n_nodes = 4, stem_multiplier = 3, auxiliary = False):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1
        c_cur = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_cur, 3, 1, 1, bias = False),
            nn.BatchNorm2d(c_cur)
        )

        # for the first cell, stem is used for both s0 and s1, channels_pp and channels_p is output channel size, c_cur is input channel size.
        channels_pp, channels_p, c_cur = c_cur, c_cur, channels
        self.cells = nn.ModuleList()
        reduction_p, reduction = False, False
        for i in range(n_layers):
            reduction_p, reduction = reduction, False
            if i in [n_layers // 3, 2* n_layers // 3]:
                c_cur *= 2
                reduction = True
            cell = Cell(n_nodes, channels_pp, channels_p, c_cur, reduction_p, reduction)
            self.cells.append(cell)
            c_cur_out = c_cur * n_nodes
            channels_pp, channels_p = channels_p, c_cur_out
            if i == self.aux_pos:
                self.aux_head = AuxiliaryHead(input_size // 4, channels_p, n_classes)
        self.gap = nn.AdaptiveAvgPool2d(input_size // 4, channels_p, n_classes)
        self.linear = nn.Linear(channels_p, n_classes)
    
    def drop_path_prob(self, p):
        for module in self.modules():
            if isinstance(module, DropPath):
                module.p = p

    def forward(self, x):
        s0 = s1 = self.stem(x)
        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)
        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        if aux_logits is not None:
            return logits, aux_logits
        return logits