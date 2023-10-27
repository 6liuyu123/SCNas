from Node import Node
from Ops import FactorizedReduce, StdConv

import torch
import torch.nn as nn

class Cell(nn.Module):
    def __init__(self, n_nodes, channels_pp, channels_p, channels, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        # If previous cell is reduction cell, current input size does not match with output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = FactorizedReduce(channels_pp, channels, affline = False)
        else:
            self.preproc0 = StdConv(channels_pp, channels, 1, 1, 0, affline = False)
        self.preproc1 = StdConv(channels_p, channels, 1, 1, 0, affline = False)
        # generate DAG
        self.mutable_ops = nn.ModuleList()
        for depth in range(2, self.nodes + 2):
            self.mutable_ops.append(Node("{}_n{}".format("reduce" if reduction else "normal", depth), depth, channels, 2 if reduction else 0))
                                                         
    def forward(self, s0, s1):
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.mutable_ops:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)
        output = torch.cat(tensors[2:])
        return output