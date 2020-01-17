import torch


class LinkNode(object):
    def __init__(self, name):
        self.name = name
        self.up = []
        self.down = []
        self.opc = []
        self.ipc = []


def backtracking(tree_dict, key_number, key_list):
    return_list = []
    for in_item in tree_dict[key_number]:
        if in_item in key_list or in_item == 0:
            return_list += [in_item]
        else:
            return_list += backtracking(tree_dict, in_item, key_list)
    return return_list


def list_network_nodes(network, data):
    trace, _ = torch.jit.get_trace_graph(network, data)
    torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    return list(trace.graph().nodes())


if __name__ == "__main__":
    import torchvision.models as models

    support_module_list = ('onnx::Conv', 'onnx::BatchNormalization',)
    demo_data = torch.zeros([1, 3, 224, 224]).cuda()
    demo_net = models.resnet18().cuda()

    node_list = list_network_nodes(demo_net, demo_data)

    node_index2name = {}
    node_index_link = {}
    unit_index_link = {}

    for node in node_list:
        node_name_list = node.scopeName().split('/')[1:]
        node_name = ''
        for sub_name in node_name_list:
            node_name += '.' + sub_name.split('[')[-1][:-1]
        node_name = node_name
        print(node_name)
        # node_index = node.outputs().__next__().unique()
        # node_index2name[node_index] = node_name
        # node_index_link[node_index] = []
        # for input_node in node.inputs():
        #     input_index = input_node.unique()
        #     if input_index == 0 or input_index in node_index_link.keys():
        #         node_index_link[node_index].append(input_index)
        # if node.kind() in support_module_list:
        #     layer_index_link[node_index] = backtracking(node_index_link, node_index, layer_index_link.keys())
