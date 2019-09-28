import torch
import torch.nn as nn

from GateDecorator.GatedBN import GatedBN


class GatedNet(nn.Module):
    def __init__(self, model_ipc=3):
        super().__init__()

        def conv(ipc, opc, stride=1):
            return nn.ModuleList([nn.Conv2d(ipc, opc, 3, stride=stride, padding=1, bias=False), GatedBN(opc)])

        self.conv_trunk = nn.Sequential(conv(model_ipc, 4), conv(4, 8, 2), conv(8, 16),
                                        conv(16, 32), conv(32, 64, 2), conv(64, 128),
                                        conv(128, 256), conv(256, 512, 2), conv(512, 1024))
        self.linear = nn.Linear(4 * 4 * 1024, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for conv in self.conv_trunk:
            x = self.relu(conv[1](conv[0](x)))
        x = self.linear(x.reshape(x.size(0), -1))
        return x

    def freeze(self):
        for conv in self.conv_trunk:
            conv[1].freeze()

    def melt(self):
        for conv in self.conv_trunk:
            conv[1].melt()

    def prune(self):
        for conv in self.conv_trunk:
            print(conv[1].prune())


if __name__ == "__main__":
    from GateDecorator.Train import train_model, valid_model

    gated_net = GatedNet(1)
    train_model(gated_net, "gated_train.pkl", batch_size=6000, epochs=3)
    valid_model(gated_net, "gated_train.pkl", batch_size=10000)

    gated_net.freeze()
    # gated_net.load_state_dict(torch.load("gated_freeze.pkl"))
    train_model(gated_net, "gated_freeze.pkl", batch_size=6000, epochs=1, lr=1e-4)
    gated_net.prune()
    valid_model(gated_net, "gated_train.pkl", batch_size=10000)

    # gated_net.melt()
    # valid_model(gated_net, "gated_train.pkl", batch_size=10000)
    #
    # train_model(gated_net, "gated_melt.pkl", batch_size=6000, epochs=1, lr=1e-5)
    # valid_model(gated_net, "gated_train.pkl", batch_size=10000)
