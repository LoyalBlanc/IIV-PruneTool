import numpy as np
import old_version.Module.basic_module as bm
import torch.nn as nn
from old_version.utils import utils


class AbstractNetwork(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.layer_trunk = None
        self.link_matrix = None
        self.prune_relationship = None
        self.hook = None
        self.regularization = 0

    def forward(self, x):
        features = []
        for layer_index, layer in enumerate(self.layer_trunk):
            feature = x if layer_index == 0 else \
                np.sum([features[link] for link in utils.get_related_row(self.link_matrix, layer_index)])
            features.append(layer(feature))
            # print(layer(feature).shape)
        return features[-1]

    def link_matrix_analysis(self):
        self.link_matrix = np.tril(self.link_matrix, k=-1)
        self.prune_relationship = []
        for layer_index in range(len(self.layer_trunk)):
            self.prune_relationship.append(utils.list_extend_row_col(self.link_matrix, layer_index))

    def prune(self, layer_index, channel_index):
        if self.prune_relationship is None:
            self.link_matrix_analysis()
        prune_opc_list, prune_ipc_list = self.prune_relationship[layer_index]
        for prune_opc_index in prune_opc_list:
            self.layer_trunk[prune_opc_index].prune_opc(channel_index)
        for prune_ipc_index in prune_ipc_list:
            self.layer_trunk[prune_ipc_index].prune_ipc(channel_index)


class TestNet(AbstractNetwork):
    def __init__(self):
        AbstractNetwork.__init__(self)
        self.layer_trunk = nn.Sequential(bm.BasicModule(4, 5, 3), bm.BasicModule(5, 5, 3),
                                         bm.BasicModule(5, 5, 3), bm.Conv2d(5, 4, 3, padding=1))

        self.link_matrix = np.array([[0, 0, 0, 0],
                                     [1, 0, 0, 0],
                                     [1, 1, 0, 0],
                                     [1, 1, 1, 0]])


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
