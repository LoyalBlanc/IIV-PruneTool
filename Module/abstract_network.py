import numpy as np
import torch.nn as nn
from utils import get_related_row, get_extend_row_col
from Module.basic_module import BasicModule, Conv2d


class AbstractNetwork(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv_trunk = None
        self.skip_mat = None

    def forward(self, x):
        features = []
        for conv_index, conv in enumerate(self.conv_trunk):
            feature = x if conv_index == 0 else \
                np.sum([features[link] for link in get_related_row(self.skip_mat, conv_index)])
            features.append(conv(feature))
        return x

    def prune(self, layer_index, channel_index):
        prune_opc_list, prune_ipc_list = get_extend_row_col(self.skip_mat, layer_index)
        for prune_opc_index in prune_opc_list:
            self.conv_trunk[prune_opc_index].prune_opc(channel_index)
        for prune_ipc_index in prune_ipc_list:
            self.conv_trunk[prune_ipc_index].prune_ipc(channel_index)


class TestNet(AbstractNetwork):
    def __init__(self):
        AbstractNetwork.__init__(self)
        self.conv_trunk = nn.Sequential(BasicModule(4, 5, 3), BasicModule(5, 5, 3),
                                        BasicModule(5, 5, 3), Conv2d(5, 4, 3, padding=1))

        self.skip_mat = np.tril(np.array([[0, 0, 0, 0],
                                          [1, 0, 0, 0],
                                          [1, 1, 0, 0],
                                          [1, 1, 1, 0]]), k=-1)


if __name__ == "__main__":
    from torch import rand, Size

    test_data = rand(1, 4, 16, 16)
    test_net = TestNet()
    assert test_net(test_data).shape == Size([1, 4, 16, 16])

    test_net.prune(0, 1)
    test_net.prune(1, 2)
    test_net.prune(2, 3)
    test_net.prune(3, 3)
    assert test_net(test_data).shape == Size([1, 4, 16, 16])

    print("Pass the unit exam!")
