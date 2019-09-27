import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()

        def conv(ipc, opc, stride=1):
            return nn.Sequential(nn.Conv2d(ipc, opc, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(opc))

        self.conv_trunk = nn.Sequential(conv(1, 4), conv(4, 16, 2), conv(16, 64, 2), conv(64, 256, 1))
        self.linear = nn.Linear(7 * 7 * 256, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for conv in self.conv_trunk:
            x = self.relu(conv(x))
        x = self.linear(x.reshape(x.size(0), -1))
        return x


if __name__ == "__main__":
    from GateDecorator.Train import train_model, valid_model

    baseline = Baseline()
    train_model(baseline, "baseline.pkl", batch_size=10000, epochs=10)
    valid_model(baseline, "baseline.pkl", batch_size=10000)
