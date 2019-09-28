import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()

        def conv(ipc, opc, stride=1):
            return nn.Sequential(nn.Conv2d(ipc, opc, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(opc))

        self.conv_trunk = nn.Sequential(conv(3, 4), conv(4, 8, 2), conv(8, 16),
                                        conv(16, 32), conv(32, 64, 2), conv(64, 128),
                                        conv(128, 256), conv(256, 512, 2), conv(512, 1024))
        self.linear = nn.Linear(4 * 4 * 1024, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for conv in self.conv_trunk:
            x = self.relu(conv(x))
        x = self.linear(x.reshape(x.size(0), -1))
        return x


if __name__ == "__main__":
    from GateDecorator.Train import train_model, valid_model

    baseline = Baseline()
    train_model(baseline, "baseline.pkl", batch_size=6000, epochs=10)
    valid_model(baseline, "baseline.pkl", batch_size=10000)
