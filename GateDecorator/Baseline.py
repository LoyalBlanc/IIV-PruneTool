import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 4, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(4))
        self.linear = nn.Linear(7 * 7 * 4, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    from GateDecorator.Train import train_model

    baseline = Baseline()
    train_model(baseline)
