from argparse import ArgumentParser
from Net import Net

import datasets
import torch
import torch.nn as nn

if __name__ == "__main__":
    parser = ArgumentParser("SCNas")
    parser.add_argument("--layers", default = 8, type = int)
    parser.add_argument("--batch-size", default = 64, type = int)
    parser.add_argument("--log-frequency", default = 10, type = int)
    parser.add_argument("--epochs", default = 50)
    parser.add_argument("--channels", default = 16, type = int)
    parser.add_argument("--unrolled", default = False, action = "store_true")
    parser.add_argument("--visualization", default = False, action = "store_true")
    args = parser.parse_args()
    dataset_train, dataset_valid = datasets.get_dataset("cifar10")

    """
        Class: Net
        Args: input_size, in_channels, channels, n_classes, n_layers
    """
    model = Net(32, 3, args.channels, 10, args.layers)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), 0.025, momentum = 0.9, weight_decay = 3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min = 0.001)
    trainer = Trainerv1(model, loss)