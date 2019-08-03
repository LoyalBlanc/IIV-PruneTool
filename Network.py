from Module import *


class PruningNetwork(nn.Module):
    def __init__(self, dim_list):
        super(PruningNetwork, self).__init__()
        self._conv = nn.ModuleList()
        for dim_index in range(len(dim_list) - 1):
            self._conv.append(PruningConvolution(dim_list[dim_index], dim_list[dim_index + 1]))
        self.max = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        for conv in self._conv:
            x = conv(x)
        po = self.max(x).reshape(-1, 10)
        return po

    def prune(self, iteration=1):
        # Todo: Pruning sensitivity / Fixed quantity / Fixed percentage
        score = []
        for conv in self._conv[:-1]:
            conv.calculate_channel_contribution()
            score.append(conv.pruned_channel_score)
        for _ in range(iteration):
            prune_index = score.index(min(score))
            self._conv[prune_index + 1].prune_ipc(self._conv[prune_index].prune_opc())
            self._conv[prune_index].calculate_channel_contribution()
            score[prune_index] = self._conv[prune_index].pruned_channel_score

    def get_new_list(self):
        # Returns the current number of convolution channels
        return [conv.get_pruned_channel()[0] for conv in self._conv] + [self._conv[-1].get_pruned_channel()[1]]


if __name__ == "__main__":
    from Dataset import DataPreFetcher

    fetcher = DataPreFetcher(100, train=True)
    fetcher.refresh()
    image, label = fetcher.next()
    image = image.cpu()
    test_net = PruningNetwork([1, 64, 64, 64, 64, 10])
    print(test_net.get_new_list())
    print(test_net(image).shape, label.shape)
    test_net.prune()
    print(test_net.get_new_list())
    print(test_net(image).shape, label.shape)
