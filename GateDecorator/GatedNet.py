import torch
import torch.nn as nn
from GateDecorator.GatedModule import GbnModule


class GatedNet(nn.Module):
    def __init__(self, model_ipc):
        super().__init__()
        self.conv_trunk = nn.ModuleList([GbnModule(model_ipc, 4), GbnModule(4, 8, 2), GbnModule(8, 16),
                                         GbnModule(16, 32), GbnModule(32, 64, 2), GbnModule(64, 128),
                                         GbnModule(128, 256), GbnModule(256, 512, 2), GbnModule(512, 1024)])
        self.linear = nn.Linear(4 * 4 * 1024, 10)

    def forward(self, x):
        for conv in self.conv_trunk:
            x = conv(x)
        return self.linear(x.reshape(x.size(0), -1))

    def to_gbn(self):
        for conv in self.conv_trunk:
            conv.to_gbn()

    def freeze(self):
        for conv in self.conv_trunk:
            conv.freeze()

    def prune(self, threshold=16):
        prune_score = []
        for i_conv, conv in enumerate(self.conv_trunk):
            for i_channel, score in enumerate(conv.score.squeeze()):
                prune_score.append((score, i_channel, i_conv))
            conv.score = 0
        prune_score = torch.Tensor(prune_score)
        for index in prune_score[:, 0].sort(0)[1][0:threshold]:
            _, i_channel, i_conv = prune_score[index]
            if i_conv < 8:
                self.conv_trunk[int(i_conv)].prune_opc(int(i_channel))
                self.conv_trunk[int(i_conv) + 1].prune_ipc(int(i_channel))

    def melt(self):
        for conv in self.conv_trunk:
            conv.melt()

    def to_bn(self):
        for conv in self.conv_trunk:
            conv.to_bn()


if __name__ == "__main__":
    from GateDecorator.Train import train_model, valid_model

    gated_net = GatedNet(1)
    test_batch_size = 10000

    train_model(gated_net, batch_size=test_batch_size, epochs=10, lr=1e-4)
    valid_model(gated_net)

    gated_net.to_gbn()
    for _ in range(5):
        gated_net.freeze()
        train_model(gated_net, batch_size=test_batch_size, epochs=1, lr=1e-5)
        gated_net.prune()
        valid_model(gated_net)
        gated_net.melt()
        train_model(gated_net, batch_size=test_batch_size, epochs=1, lr=1e-6)
        valid_model(gated_net)
    gated_net.to_bn()
    valid_model(gated_net)
