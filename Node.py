from collections import OrderedDict
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
from Ops import DilConv, DropPath, FactorizedReduce, PoolBN, SepConv

import torch.nn as nn

class Node(nn.Module):
    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_connect):
        super().__init__()
        self.ops = nn.ModuleList()
        choice_keys = []
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            choice_keys.append("{}_p{}".format(node_id, i))
            self.ops.append(
                LayerChoice(OrderedDict([
                    ("maxpool", PoolBN('max', channels, 3, stride, 1, affine = False)),
                    ("avgpool", PoolBN('avg', channels, 3, stride, 1, affine = False)),
                    ("skipconnect", nn.Identity() if stride == 1 else FactorizedReduce(channels, channels, affine = False)),
                    ("sepconv3x3", SepConv(channels, channels, 3, stride, 1, affine = False)),
                    ("sepconv5x5", SepConv(channels, channels, 5, stride, 2, affine = False)),
                    ("dilconv3x3", DilConv(channels, channels, 3, stride, 2, 2, affine = False)),
                    ("dilconv5x5", DilConv(channels, channels, 5, stride, 4, 2, affine = False))
                ]), label = choice_keys[-1])
            )
        self.drop_path = DropPath()
        self.input_switch = InputChoice(n_candidates = len(choice_keys), n_chosen = 2, label = "{}_switch".format(node_id))
    
    def forward(self, prev_nodes):
        assert len(self.ops) == len(prev_nodes)
        out = [op(node) for op, node in zip(self.ops, prev_nodes)]
        out = [self.drop_path(0) if o is not None else None for o in out]
        return self.input_switch(out)
