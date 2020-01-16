import torch


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
    detect_list = ('onnx::Conv', 'onnx::BatchNormalization')

    node_index2name = {}
    node_index_link = {}
    layer_index_link = {}

    for node in node_list:
        node_name = node.scopeName().split('[')[-1][:-1]
        node_index = node.outputs().__next__().unique()
        node_index2name[node_index] = node_name
        node_index_link[node_index] = []
        for input_node in node.inputs():
            input_index = input_node.unique()
            if input_index == 0 or input_index in node_index_link.keys():
                node_index_link[node_index].append(input_index)
        if node.kind() in detect_list:
            layer_index_link[node_index] = backtracking(node_index_link, node_index, layer_index_link.keys())

    return node_index2name, layer_index_link
