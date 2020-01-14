import torch


def find_minimum_weight(network):
    layer_score = [torch.norm(channel_weight, p=2).item() for channel_weight in network.layer_trunk[0].conv.weight]
    minimum_weight = min(layer_score)
    layer_index = 0
    channel_index = layer_score.index(minimum_weight)
    for index, layer in enumerate(network.layer_trunk[1:-1]):
        layer_score = [torch.norm(channel_weight, p=2).item() for channel_weight in layer.conv.weight]
        temp_minimum_weight = min(layer_score)
        if temp_minimum_weight < minimum_weight:
            minimum_weight = temp_minimum_weight
            layer_index = index + 1
            channel_index = layer_score.index(minimum_weight)
    return layer_index, channel_index, minimum_weight


def prune_one_channel(network):
    layer_index, channel_index, minimum_weight = find_minimum_weight(network)
    network.prune(layer_index, channel_index)
    print("Prune Channel %d in Layer %d (Weight %.4f)." % (layer_index, channel_index, minimum_weight))


if __name__ == "__main__":
    from torch import rand, Size
    from Module.abstract_network import TestNet

    test_data = rand(1, 4, 16, 16)
    test_net = TestNet()
    assert test_net(test_data).shape == Size([1, 4, 16, 16])

    test_layer_index, test_channel_index, _ = find_minimum_weight(test_net)
    test_net.prune(test_layer_index, test_channel_index)
    assert test_net(test_data).shape == Size([1, 4, 16, 16])

    print("Pass the unit exam!")
