import torch
import torch.nn as nn
from Network.AbstractNetwork import AbstractNetwork


class ConvNet(AbstractNetwork):
    def __init__(self, module):
        super().__init__()
        self.conv_trunk.extend([module(1, 4), module(4, 4, 2), module(4, 8), module(8, 8, 2), module(8, 16)])

    def forward(self, x):
        x = super().forward(x)
        return x

    def calculate_network_contribution(self):
        self.contribution = torch.Tensor()
        self.contribution_index = torch.Tensor()
        for conv_index, conv in enumerate(self.conv_trunk):
            conv.calculate_channel_contribution()
            self.contribution = torch.cat((self.contribution, conv.get_channel_contribution()), dim=0)
            temp_index = torch.Tensor([(conv_index, channel_index) for channel_index in range(conv.opc)])
            self.contribution_index = torch.cat((self.contribution_index, temp_index), dim=0)

    def prune_index(self, conv_index, channel_index):
        self.conv_trunk[conv_index].prune_opc(channel_index)
        self.conv_trunk[conv_index + 1].prune_ipc(channel_index)


if __name__ == "__main__":
    from Network.AbstractNetwork import network_test
    from Module.MinimumWeight.MinimumWeight import Basic

    network_test(ConvNet, Basic, ipc=1, opc=16, stride=4, channels=40, depth=5)
