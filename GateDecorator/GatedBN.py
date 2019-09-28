import torch
import torch.nn as nn


class GatedBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.bn = nn.BatchNorm2d(self.channels)
        self.fai = nn.Parameter(torch.ones(1, self.channels, 1, 1), requires_grad=False)
        self.score = 0
        self.hook = None

    def forward(self, x):
        return self.fai * self.bn(x)

    def freeze(self):
        with torch.no_grad():
            self.fai.set_(self.fai * self.bn.weight.view(1, -1, 1, 1))
            self.fai.requires_grad = True
            self.bn.bias.set_(torch.clamp(self.bn.bias / self.bn.weight, -10, 10))
            self.bn.weight.set_(torch.ones_like(self.bn.weight))
            self.bn.weight.requires_grad = False
        self.hook = self.fai.register_hook(self.calculate_score)

    def melt(self):
        with torch.no_grad():
            self.bn.bias.set_(self.bn.bias * self.fai)
            self.bn.weight.set_(self.fai)
            self.bn.weight.requires_grad = True
            self.fai = nn.Parameter(torch.ones(1, self.channels, 1, 1), requires_grad=False)
        self.hook.remove()
        self.hook = None

    def calculate_score(self, grad):
        self.score += (grad * self.fai).abs()


if __name__ == "__main__":
    pass
