import torch
import torch.nn as nn
from Frame.AbstractFrame import AbstractFrame


class OneShot(AbstractFrame):
    def __init__(self, prune_ratio):
        super().__init__()
        self.prune_ratio = prune_ratio

    def prune(self, network):
        network.calculate_network_contribution()
        network_score, network_score_index = network.get_network_contribution()
        prune_limit = int(self.prune_ratio * network_score.shape[0])
        prune_index = network_score_index[network_score.sort(0)[1][:prune_limit]]
        for conv_index, channel_index in prune_index.sort(0, descending=True)[0]:
            network.prune_index(conv_index, channel_index)


if __name__ == "__main__":
    from Network.ConvNet import ConvNet
    from Module.MinimumWeight.MinimumWeight import Basic

    torch.manual_seed(229)
    test_prune = OneShot(0.2)
    test_network = ConvNet(Basic)
    print(test_network.get_pruned_channel())

    test_prune.prune(test_network)
    print(test_network.get_pruned_channel())
