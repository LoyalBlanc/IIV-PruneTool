import torch
from Frame.AbstractFrame import AbstractFrame


class OneShot(AbstractFrame):
    def __init__(self, prune_ratio):
        super().__init__()
        self.prune_ratio = prune_ratio

    def prune(self, network):
        network.before_pruning_network()
        network.calculate_network_contribution()
        network_score, network_score_index = network.get_network_contribution()
        prune_limit = int(self.prune_ratio * network_score.shape[0])
        prune_index = network_score_index[network_score.sort(dim=0)[1][:prune_limit]]
        _, prune_index_sorted = prune_index[:, 1].sort(dim=0, descending=True)
        for index in prune_index_sorted:
            conv_index, channel_index = prune_index[index]
            network.prune_index(conv_index, channel_index)
        network.after_pruning_network()


if __name__ == "__main__":
    from Demo.Model import DemoNetworkForPruning
    from Module.MinimumWeight.MinimumWeight import Basic

    torch.manual_seed(229)
    test_prune = OneShot(0.5)
    test_network = DemoNetworkForPruning(Basic)
    print(test_network.get_pruned_channel())
    test_prune.prune(test_network)
    print(test_network.get_pruned_channel())
