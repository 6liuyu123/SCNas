from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeterGroup
from SCNasMutator import SCNasMutator

import copy
import torch

class Trainerv1(Trainer):
    def __init__(self, model, loss, metrics, optimizer, num_epochs, dataset_train, dataset_valid, mutator = None, batch_size = 64,
                 workers = 4, device = None, log_frequency = None, callbacks = None, arc_learning_rate= 3.0E-4, unrolled = False):
        super().__init__(model, mutator if mutator is not None else SCNasMutator)
        self.ctrl_optim = torch.optim.Adam(self.mutator.parameters(), arc_learning_rate, betas = (0.5, 0.999), weight_decay = 1.0E-3)
        self.unrolled = unrolled
        n_train = len(self.dataset_train)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size = batch_size, sampler = train_sampler, num_workers = workers)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_valid, batch_size = batch_size, sampler = valid_sampler, num_workers = workers)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size = batch_size, num_workers = workers)

    def train_one_epoch(self, epoch):
        self.model.train()
        self.mutator.train()
        meters = AverageMeterGroup()
        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(self.train_loader, self.valid_loader)):
            trn_X, trn_y = trn_X.to(self.device), trn_y.to(self.device)
            val_X, val_y = val_X.to(self.device), val_y.to(self.device)
            self.ctrl_optim.zero_grad()
            if self.unrolled:
                self._unrolled_backward(trn_X, trn_y, val_X, val_y)
    
    """
        Compute unrolled weights w
        Don't need zero_grad, using autograd to calculate gradients
    """
    def _compute_virtual_model(self, X, y, lr, momentum, weight_decay):
        _, loss = self._logits_and_loss(X, y)

    """
        Compute unrolled loss and backward its gradients
    """
    def _unrolled_backward(self, trn_X, trn_y, val_X, val_y):
        backup_params = copy.deepcopy(tuple(self.model.parameters()))
        lr = self.optimizer.param_groups["0"]["lr"]
        momentum = self.optimizer.param_groups["0"]["momentum"]
        weight_decay = self.optimizer.param_groups["0"]["weight_decay"]
        self._compute_virtual_model(trn_X, trn_y, lr, momentum, weight_decay)
