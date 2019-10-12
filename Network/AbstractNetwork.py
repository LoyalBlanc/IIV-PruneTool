import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class AbstractNetwork(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()
        self.conv_trunk = nn.ModuleList()

        self.contribution = torch.Tensor()
        self.contribution_index = torch.IntTensor()
        self.hook = None
        self.regularization = 0

    def forward(self, x):
        for conv in self.conv_trunk:
            x = conv(x)
        return x

    def before_pruning_network(self):
        def calculate_regularization(network, input_tensor, output_tensor):
            network.regularization = 0
            for module in network.conv_trunk:
                network.regularization += module.regularization

        self.hook = self.register_forward_hook(calculate_regularization)
        for conv in self.conv_trunk:
            conv.before_pruning_module()

    def after_pruning_network(self):
        self.hook.remove()
        self.regularization = 0
        for conv in self.conv_trunk:
            conv.after_pruning_module()

    @abstractmethod
    def calculate_network_contribution(self):
        pass

    def get_network_contribution(self):
        return self.contribution, self.contribution_index

    @abstractmethod
    def prune_index(self, conv_index, channel_index):
        pass

    def get_pruned_channel(self):
        return torch.IntTensor([conv.opc for conv in self.conv_trunk])


def network_test(network, module,
                 batch_size=1, ipc=2, data_size=32,
                 output_size=torch.Size([1, 10]), channels=255, depth=9):
    test_network = network(module=module)
    print("Origin:\nchannels:", test_network.get_pruned_channel())
    test_network.before_pruning_network()

    test_data = torch.randn(batch_size, ipc, data_size, data_size)
    test_output = test_network(test_data)
    assert test_output.shape == output_size
    test_network.calculate_network_contribution()
    test_score, test_score_index = test_network.get_network_contribution()
    print("score:{}\nregularization:{}\n".format(test_score, test_network.regularization))
    assert test_score_index.shape == torch.Size([channels, 2])

    for conv_index in range(depth):
        for channel_index in range(2):
            test_network.prune_index(conv_index, channel_index)
    print("Pruned:\nchannels:", test_network.get_pruned_channel())
    test_network.after_pruning_network()

    test_output = test_network(test_data)
    assert test_output.shape == output_size
    test_network.calculate_network_contribution()
    test_score, test_score_index = test_network.get_network_contribution()
    print("score:{}\nregularization:{}\n".format(test_score, test_network.regularization))
    assert test_score_index.shape == torch.Size([channels - 2 * depth, 2])
