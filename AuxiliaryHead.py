import torch.nn as nn

"""
    Auxiliary head in 2/3 place of network to let the gradient flow well
"""
class AuxiliaryHead(nn.Module):
    def __init__(self, input_size, C, n_classes):
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.AvgPool2d(5, stride = input_size - 5, padding = 0, count_including_pad = False), # 2x2 out
            nn.Conv2d(C, 128, kernel_size = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 768, kernel_size = 2, bias = False), # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace = True)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits