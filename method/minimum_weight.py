import torch


def before_pruning(network):
    module_dict = network.get_modules()
    for module_name in module_dict:
        module = module_dict[module_name]
        if hasattr(module, "connect_flag"):
            if not module.connect_flag and not network.layer_name_link[module_name].down == []:
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


def prune_once(network):
    modules_score = {}
    for module_name in network.prune_score_range:
        module_current = eval("network." + module_name)
        module_next = eval("network." + network.layer_name_link[module_name].down[0])

        layer_norm = torch.Tensor([torch.norm(channel_weight, p=2).item() for channel_weight in module_current.weight])
        layer_bias = module_next.bias
        layer_weight = module_next.weight
        layer_score = (layer_norm - layer_bias) * layer_weight

        modules_score[module_name] = layer_score

    minimum_score = None
    layer_name = None
    channel_index = None
    for module_name in network.prune_score_range:
        temp_score = None
        for item in network.layer_name_link[module_name].opc:
            temp_score = modules_score[item] if temp_score is None else temp_score + modules_score[item]
            temp_minimum_score = min(temp_score).item()
            if minimum_score is None:
                minimum_score = temp_minimum_score
                layer_name = module_name
                channel_index = torch.where(temp_score == temp_minimum_score)[0].item()
            elif minimum_score > temp_minimum_score:
                minimum_score = temp_minimum_score
                layer_name = module_name
                channel_index = torch.where(temp_score == temp_minimum_score)[0].item()

    layer_info = network.prune(layer_name, channel_index)
    print("Pruning Channel %d in %s (Score %.4f)" % (channel_index, layer_info, minimum_score))


if __name__ == "__main__":
    from demo import DemoNet

    demo_net = DemoNet()
    before_pruning(demo_net)
    prune_once(demo_net)
