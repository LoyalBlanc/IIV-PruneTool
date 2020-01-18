import copy

import torch


def prepare_pruning(network, example_data):
    unit_dict = network.unit_dict
    network.pruning_range = []
    for unit_name in unit_dict.keys():
        if eval("network" + unit_name).is_layer:
            network.pruning_range.append(unit_name)

    # Todo: Here must provide a beautiful method!
    del_list = []
    for unit_name in network.pruning_range:
        network_backup = copy.deepcopy(network)
        network_backup.prune_spec_channel(unit_name, 0)
        try:
            _ = network_backup(example_data)
        except RuntimeError:
            del_list.append(unit_name)
    for unit_name in del_list:
        network.pruning_range.remove(unit_name)


def prune_network_once(network):
    unit_dict = network.unit_dict
    unit_score = {}
    # channel score = (conv.weight - bn.bias) * bn.weight
    for unit_name in network.pruning_range:
        unit_current = eval("network" + unit_name)
        unit_next = eval("network" + unit_dict[unit_name].next[0])
        layer_norm = torch.Tensor([torch.norm(channel_weight, p=2).item() for channel_weight in unit_current.weight])
        layer_bias = unit_next.bias
        layer_weight = unit_next.weight
        layer_score = (layer_norm - layer_bias) * layer_weight
        unit_score[unit_name] = layer_score

    # calculate and find the least related score
    minimum_score = None
    pruning_unit_name = None
    channel_index = None
    for unit_name in network.pruning_range:
        temp_score = None
        for item in unit_dict[unit_name].affect_opc:
            temp_score = unit_score[item] if temp_score is None else temp_score + unit_score[item]
            temp_minimum_score = min(temp_score).item()
            if minimum_score is None:
                minimum_score = temp_minimum_score
                pruning_unit_name = unit_name
                channel_index = torch.where(temp_score == temp_minimum_score)[0].item()
            elif minimum_score > temp_minimum_score:
                minimum_score = temp_minimum_score
                pruning_unit_name = unit_name
                channel_index = torch.where(temp_score == temp_minimum_score)[0].item()

    layer_info = network.prune_spec_channel(pruning_unit_name, channel_index)
    print("Pruning Channel %d in %s (Score %.4f)" % (channel_index, layer_info, minimum_score))


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


if __name__ == "__main__":
    import torchvision.models as models
    from pruning_tools import network_analyzer as nwa

    model = models.resnet18()
    data = torch.ones(1, 3, 224, 224)
    nwa.analyze_network(model, data, verbose=False, for_pruning=True)
    prepare_pruning(model, data)

    for test_unit_name in model.pruning_range:
        model_backup = copy.deepcopy(model)
        model_backup.prune_spec_channel(test_unit_name, 0)
        try:
            _ = model_backup(data)
        except Exception as e:
            print(test_unit_name, e)

    prune_network_once(model)
    print(model(data).shape)
