import torch


def before_pruning(network):
    module_dict = network.get_modules()
    for module_name in module_dict:
        module = module_dict[module_name]
        if hasattr(module, "connect_flag"):
            if not module.connect_flag:
                network.prune_score_range.append(module_name)


def before_training(network):
    def conv_hook(net, input_tensor, output_tensor):
        for module_name in network.prune_score_range:
            module = eval("network." + module_name)
            net.regularization += torch.norm(module.weight, p=1)

    network.hook = network.register_forward_hook(conv_hook)
    network.regularization = 0


def after_training(network):
    network.hook.remove()
    network.regularization = 0


def prune_one_channel(network):
    minimum_weight = None
    minimum_layer_name = None
    channel_index = None
    for module_name in network.prune_score_range:
        module = eval("network." + module_name)
        layer_score = [torch.norm(channel_weight, p=2).item() for channel_weight in module.weight]
        temp_minimum_weight = min(layer_score)
        if minimum_weight is None:
            minimum_weight = temp_minimum_weight
            minimum_layer_name = module_name
            channel_index = layer_score.index(minimum_weight)
        elif temp_minimum_weight < minimum_weight:
            minimum_weight = temp_minimum_weight
            minimum_layer_name = module_name
            channel_index = layer_score.index(minimum_weight)
    network.prune(minimum_layer_name, channel_index)
    print("Pruning Channel %d in Module %s (Weight %.4f)" % (channel_index, minimum_layer_name, minimum_weight))


if __name__ == "__main__":
    from demo import DemoNet

    example_data = torch.rand(1, 5, 32, 32)
    demo_net = DemoNet()
    before_pruning(demo_net)
    cl = prune_one_channel(demo_net)
    before_training(demo_net)
    demo_net(example_data)
    after_training(demo_net)
