import torch
import torch.nn as nn


def network_analysis(network, example_data=torch.zeros([1, 1, 32, 32])):
    def backtracking(tree_dict, key_number, key_list):
        return_list = []
        for in_item in tree_dict[key_number]:
            if in_item in key_list or in_item == 0:
                return_list += [in_item]
            else:
                return_list += backtracking(tree_dict, in_item, key_list)
        return return_list

    trace, _ = torch.jit.get_trace_graph(network, example_data)
    torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    node_list = list(trace.graph().nodes())
    detect_list = ('onnx::Conv', 'onnx::BatchNormalization')
    link_dict_all = {}
    link_dict_layer = {}
    for node in node_list:
        node_index = node.outputs().__next__().unique()
        link_dict_all[node_index] = []
        for input_node in node.inputs():
            input_index = input_node.unique()
            if input_index == 0 or input_index in link_dict_all.keys():
                link_dict_all[node_index].append(input_index)
        if node.kind() in detect_list:
            link_dict_layer[node_index] = backtracking(link_dict_all, node_index, link_dict_layer.keys())
    layer_index = list(link_dict_layer.keys())
    return [[-1 if item == 0 else layer_index.index(item) for item in link_dict_layer[key]] for key in link_dict_layer]


def get_model_flops(network, data):
    network.eval()
    flops = []
    child_hook = []

    def conv_hook(self, input_tensor, output_tensor):
        batch_size, _, _, _ = input_tensor[0].size()
        output_channels, output_height, output_width = output_tensor[0].size()
        flops.append(batch_size * output_channels * output_height * output_width)

    def bn_hook(self, input_tensor, output_tensor):
        flops.append(input_tensor[0].nelement())

    def relu_hook(self, input_tensor, output_tensor):
        flops.append(input_tensor[0].nelement())

    def register(net):
        children = list(net.children())
        if not children:
            if isinstance(net, nn.Conv2d):
                child_hook.append(net.register_forward_hook(conv_hook))
            if isinstance(net, nn.BatchNorm2d):
                child_hook.append(net.register_forward_hook(bn_hook))
            if isinstance(net, nn.ReLU):
                child_hook.append(net.register_forward_hook(relu_hook))
            return
        for child in children:
            register(child)

    register(network)
    _ = network(data)
    for hook in child_hook:
        hook.remove()
    return sum(flops)
