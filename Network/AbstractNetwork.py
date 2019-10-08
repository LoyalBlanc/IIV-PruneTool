import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class AbstractNetwork(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()
        self.conv_trunk = nn.ModuleList()

        self.contribution = torch.Tensor()
        self.contribution_index = torch.Tensor()
        self.regularization = 0

    @abstractmethod
    def forward(self, x):
        for conv in self.conv_trunk:
            x = conv(x)
        return x

    @abstractmethod
    def calculate_network_contribution(self):
        pass

    def get_network_contribution(self):
        return self.contribution, self.contribution_index

    @abstractmethod
    def prune_index(self, conv_index, channel_index):
        pass


def network_test(network, module, batch_size=1, ipc=2, opc=3, data_size=32, stride=1, channels=255, depth=9):
    test_network = network(module=module)
    test_data = torch.randn(batch_size, ipc, data_size, data_size)
    test_output = test_network(test_data)
    assert test_output.shape == torch.Size([batch_size, opc, data_size // stride, data_size // stride])
    test_network.calculate_network_contribution()
    test_score, test_score_index = test_network.get_network_contribution()
    print("Origin contribution:", test_score)
    assert test_score_index.shape == torch.Size([channels, 2])

    for conv_index in range(depth - 1):
        for channel_index in range(2):
            test_network.prune_index(conv_index, channel_index)

    test_output = test_network(test_data)
    assert test_output.shape == torch.Size([batch_size, opc, data_size // stride, data_size // stride])
    test_network.calculate_network_contribution()
    test_score, test_score_index = test_network.get_network_contribution()
    print("Prune contribution:", test_score)
    assert test_score_index.shape == torch.Size([channels - 2 * (depth - 1), 2])
