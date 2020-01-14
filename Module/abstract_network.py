import numpy as np
import torch.nn as nn
from utils import get_related_row, get_extend_row_col
from Module.basic_module import BasicModule, Conv2d


class AbstractNetwork(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.layer_trunk = None
        self.link_matrix = None
        self.prune_relationship = None

    def forward(self, x):
        features = []
        for layer_index, layer in enumerate(self.layer_trunk):
            feature = x if layer_index == 0 else \
                np.sum([features[link] for link in get_related_row(self.link_matrix, layer_index)])
            features.append(layer(feature))
        return x

    def link_matrix_analysis(self):
        self.link_matrix = np.tril(self.link_matrix, k=-1)
        self.prune_relationship = []
        for layer_index in range(len(self.layer_trunk)):
            self.prune_relationship.append(get_extend_row_col(self.link_matrix, layer_index))

    def prune(self, layer_index, channel_index):
        prune_opc_list, prune_ipc_list = self.prune_relationship[layer_index]
        for prune_opc_index in prune_opc_list:
            self.layer_trunk[prune_opc_index].prune_opc(channel_index)
        for prune_ipc_index in prune_ipc_list:
            self.layer_trunk[prune_ipc_index].prune_ipc(channel_index)


class TestNet(AbstractNetwork):
    def __init__(self):
        AbstractNetwork.__init__(self)
        self.layer_trunk = nn.Sequential(BasicModule(4, 5, 3), BasicModule(5, 5, 3),
                                         BasicModule(5, 5, 3), Conv2d(5, 4, 3, padding=1))

        self.link_matrix = np.array([[0, 0, 0, 0],
                                     [1, 0, 0, 0],
                                     [1, 1, 0, 0],
                                     [1, 1, 1, 0]])
        self.link_matrix_analysis()


if __name__ == "__main__":
    from torch import rand, Size

    test_data = rand(1, 4, 16, 16)
    test_net = TestNet()
    assert test_net(test_data).shape == Size([1, 4, 16, 16])

    test_net.prune(0, 1)
    test_net.prune(1, 2)
    test_net.prune(2, 3)
    test_net.prune(3, 3)  # No effect
    assert test_net(test_data).shape == Size([1, 4, 16, 16])

    print("Pass the unit exam!")
