import torch
import torch.nn as nn
from Network.AbstractNetwork import AbstractNetwork


class BasicConv(nn.Module):
    def __init__(self, ipc, opc, stride=1):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(ipc, opc, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(opc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DemoNetworkForTraining(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.conv_trunk = nn.ModuleList([module(1, 64, 2), module(64, 64, 2), module(64, 64, 2),
                                         module(64, 64, 2), module(64, 64, 2)])
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        for conv in self.conv_trunk:
            x = conv(x)
        return self.fc(x.reshape(x.size(0), -1))


class DemoNetworkForPruning(AbstractNetwork):
    def __init__(self, module):
        super().__init__()
        self.conv_trunk.extend([module(1, 64, 2), module(64, 64, 2), module(64, 64, 2),
                                module(64, 64, 2), module(64, 64, 2)])

        self.fc_ipc_channel = 64
        self.fc_ipc_size = 1
        self.fc = nn.Linear(self.fc_ipc_channel * self.fc_ipc_size, 10)

    def forward(self, x):
        for conv in self.conv_trunk:
            x = conv(x)
        return self.fc(x.reshape(x.size(0), -1))

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
