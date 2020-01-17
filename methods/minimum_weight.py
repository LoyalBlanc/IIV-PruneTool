import torch


def prepare_pruning(network):
    layer_name_link = network.get_layer_name_link()
    insert_pruning_node(network, network, '', layer_name_link)


def insert_pruning_node(network, node, node_name, layer_link):
    children = list(node.children())
    if not children:
        if hasattr(node, "connect_flag"):
            if not node.connect_flag and not layer_link[node_name].down == []:
                network.pruning_range.append(node_name)
    else:
        node_dict = node.list_modules()
        for child in node_dict:
            child_node = node_dict[child]
            child_name = node_name + '.' + child
            insert_pruning_node(network, child_node, child_name, layer_link)


def before_training(network):
    def conv_hook(net, input_tensor, output_tensor):
        for module_name in network.pruning_range:
            module = eval("network" + module_name)
            net.regularization += torch.norm(module.weight, p=1)

    network.hook = network.register_forward_hook(conv_hook)
    network.regularization = 0


def after_training(network):
    network.hook.remove()
    network.regularization = 0


def prune_network_once(network):
    layer_name_link = network.get_layer_name_link()
    modules_score = {}
    for module_name in network.pruning_range:
        module_current = eval("network" + module_name)
        module_next = eval("network" + layer_name_link[module_name].down[0])

        layer_norm = torch.Tensor([torch.norm(channel_weight, p=2).item() for channel_weight in module_current.weight])
        layer_bias = module_next.bias
        layer_weight = module_next.weight
        layer_score = (layer_norm - layer_bias) * layer_weight

        modules_score[module_name] = layer_score

    minimum_score = None
    layer_name = None
    channel_index = None
    for module_name in network.pruning_range:
        temp_score = None
        for item in layer_name_link[module_name].opc:
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

    layer_info = network.prune_spec_channel(layer_name, channel_index)
    print("Pruning Channel %d in %s (Score %.4f)" % (channel_index, layer_info, minimum_score))


if __name__ == "__main__":
    from demo import DemoNetwork

    demo_net = DemoNetwork()
    prepare_pruning(demo_net)
    prune_network_once(demo_net)
