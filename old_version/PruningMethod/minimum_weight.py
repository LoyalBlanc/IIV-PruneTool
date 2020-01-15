import torch

import old_version.Module.basic_module as bm


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


def before_training(network):
    def conv_hook(net, input_tensor, output_tensor):
        for module in network.layer_trunk:
            if isinstance(module, bm.BasicModule):
                net.regularization += torch.norm(module.conv.weight, p=1)
            if isinstance(module, bm.Conv2d):
                net.regularization += torch.norm(module.weight, p=1)

    network.hook = network.register_forward_hook(conv_hook)
    network.regularization = 0


def after_training(network):
    network.hook.remove()


if __name__ == "__main__":
    from torch import rand, Size
    from old_version.Module.abstract_network import TestNet

    test_data = rand(1, 4, 16, 16)
    test_net = TestNet()
    assert test_net(test_data).shape == Size([1, 4, 16, 16])

    test_layer_index, test_channel_index, _ = find_minimum_weight(test_net)
    test_net.prune(test_layer_index, test_channel_index)
    assert test_net(test_data).shape == Size([1, 4, 16, 16])

    print("Pass the unit exam!")
