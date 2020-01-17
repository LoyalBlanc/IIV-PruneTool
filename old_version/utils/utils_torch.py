import torch
import torch.nn as nn

support_module_list = (
    'onnx::Conv',
    'onnx::BatchNormalization',
)


def network_link_analysis(network, input_channel):
    def backtracking(tree_dict, key_number, key_list):
        return_list = []
        for in_item in tree_dict[key_number]:
            if in_item in key_list or in_item == 0:
                return_list += [in_item]
            else:
                return_list += backtracking(tree_dict, in_item, key_list)
        return return_list

    example_data = torch.zeros([1, input_channel, 32, 32])
    trace, _ = torch.jit.get_trace_graph(network, example_data)
    torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    node_list = list(trace.graph().nodes())
    # print(node_list)

    node_index2name = {}
    node_index_link = {}
    layer_index_link = {}

    for node in node_list:
        node_name_list = node.scopeName().split('/')[1:]
        node_name = ''
        for sub_name in node_name_list:
            node_name += '.' + sub_name.split('[')[-1][:-1]
        node_name = node_name
        node_index = node.outputs().__next__().unique()
        node_index2name[node_index] = node_name
        node_index_link[node_index] = []
        for input_node in node.inputs():
            input_index = input_node.unique()
            if input_index == 0 or input_index in node_index_link.keys():
                node_index_link[node_index].append(input_index)
        if node.kind() in support_module_list:
            layer_index_link[node_index] = backtracking(node_index_link, node_index, layer_index_link.keys())

    return node_index2name, layer_index_link


def save_param(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_param(model, save_path):
    model.load_state_dict(torch.load(save_path), strict=True)


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
