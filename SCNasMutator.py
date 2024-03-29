import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.nas.pytorch.mutator import Mutator
from nni.nas.pytorch.mutables import InputChoice, LayerChoice

_logger = logging.getLogger(__name__)

class SCNasMutator(Mutator):
    def __init__(self, model):
        super().__init__(model)
        self.choices = nn.ParameterDict()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                self.choice[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length + 1))
                print(mutable.key)

    def device(self):
        for v in self.choices.values():
            return v.device
    
    def sample_search(self):
        result = dict()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                result[mutable.key] = F.softmax(self.choice[mutable.key], dim = -1)[:-1]
            elif isinstance(mutable, InputChoice):
                result[mutable.key] = torch.ones(mutable.n_candidates, dtype = torch.bool, device = self.device())
        return result
    
    def sample_final(self):
        result = dict()
        edges_max = dict()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                max_val, index = torch.max(F.softmax(self.choices[mutable.key], dim = -1)[:-1], 0)
                edges_max[mutable.key] = max_val
                result[mutable.key] = F.one_hot(index, num_classes = len(mutable)).view(-1).bool()
        for mutable in self.mutables:
            if isinstance(mutable, InputChoice):
                if mutable.n_chosen is not None:
                    weights = []
                    for src_key in mutable.choose_from:
                        if src_key not in edges_max:
                            _logger.warning("InputChoice.NO_KEY in '%s' is weighted 0 when selecting inputs", mutable.key)
                        weights = torch.tensor(weights)
                    weights = torch.tensor(weights)
                    _, topk_edge_indices = torch.topk(weights, mutable.n_chosen)
                    selected_multihot = []
                    for i, src_key in enumerate(mutable.choose_from):
                        if i not in topk_edge_indices and src_key in result:
                            result[src_key] = torch.zeros_like(result[src_key])
                        selected_multihot.append(i in topk_edge_indices)
                    result[mutable.key] = torch.tensor(selected_multihot, dtype = torch.bool, device = self.device())
                else:
                    result[mutable.key] = torch.ones(mutable.n_candidates, dtype = torch.bool, device = self.device())
        return result