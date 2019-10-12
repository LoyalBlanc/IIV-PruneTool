import torch
import torch.nn as nn
from Network.AbstractNetwork import AbstractNetwork


class ConvNet(AbstractNetwork):
    def __init__(self, module, data_size=32):
        super().__init__()
        self.conv_trunk.extend([module(1, 4), module(4, 4, 2), module(4, 8), module(8, 8, 2), module(8, 16)])

        self.data_size = data_size
        self.fc_ipc_channel = 16
        self.fc_ipc_size = self.data_size ** 2 // 16
        self.fc = nn.Linear(self.fc_ipc_channel * self.fc_ipc_size, 10)

    def forward(self, x):
        x = super().forward(x)
        x = self.fc(x.reshape(x.size(0), -1))
        return x

    def calculate_network_contribution(self):
        self.contribution = torch.Tensor()
        self.contribution_index = torch.IntTensor()
        for conv_index, conv in enumerate(self.conv_trunk):
            conv.calculate_channel_contribution()
            self.contribution = torch.cat((self.contribution, conv.get_channel_contribution()), dim=0)
            temp_index = torch.IntTensor([(conv_index, channel_index) for channel_index in range(conv.opc)])
            self.contribution_index = torch.cat((self.contribution_index, temp_index), dim=0)

    def prune_index(self, conv_index, channel_index):
        self.conv_trunk[conv_index].prune_opc(channel_index)
        if conv_index != 4:
            self.conv_trunk[conv_index + 1].prune_ipc(channel_index)
        else:
            self.fc_ipc_channel -= 1
            fc_weight = torch.cat((self.fc.weight[:, 0:channel_index * self.fc_ipc_size],
                                   self.fc.weight[:, (channel_index + 1) * self.fc_ipc_size:]), dim=1)
            fc_bias = self.fc.bias
            self.fc = nn.Linear(self.fc_ipc_channel * self.fc_ipc_size, 10)
            self.fc.weight = nn.Parameter(fc_weight)
            self.fc.bias = nn.Parameter(fc_bias)


if __name__ == "__main__":
    from Network.AbstractNetwork import network_test
    from Module.MinimumWeight.MinimumWeight import Basic

    network_test(ConvNet, Basic, ipc=1, channels=40, depth=5, data_size=32)
