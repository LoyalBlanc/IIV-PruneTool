import torch

SUPPORT_LAYER_TUPLE = ('onnx::Conv',)
SUPPORT_NON_LAYER_TUPLE = ('onnx::BatchNormalization',)


class LinkNode(object):
    def __init__(self, name):
        self.name = name
        self.previous = []
        self.next = []
        self.affect_opc = []
        self.affect_ipc = []

    def __repr__(self):
        repr_info = "Node " + self.name + ': [' + ",".join(self.previous) + "] -> " \
                    + self.name + "(This Node) -> [" + ",".join(self.next) \
                    + "]\nPruning affect:\n\topc: [" + ",".join(self.affect_opc) \
                    + "]\n\tipc: [" + ",".join(self.affect_ipc) + "]\n"
        return repr_info


def _backtracking(tree_dict, key_number, key_list):
    return_list = []
    for in_item in tree_dict[key_number]:
        if in_item in key_list or in_item == 0:
            return_list += [in_item]
        else:
            return_list += _backtracking(tree_dict, in_item, key_list)
    return return_list


def _get_node_name(describe):
    describe_list = describe.split('.')[:-1]
    name = ''
    for item in describe_list:
        name += '[' + item + ']' if item.isdigit() else '.' + item
    return name


def _insert_unit(network, container_node, object_name, unit_name_dict, target_place=0):
    if not eval("network" + object_name).is_layer:
        for unit in unit_name_dict[object_name].next:
            _insert_unit(network, container_node, unit, unit_name_dict, 1)
        for unit in unit_name_dict[object_name].previous:
            _insert_unit(network, container_node, unit, unit_name_dict, 0)

    elif target_place == 0:
        if object_name not in container_node.affect_opc:
            container_node.affect_opc.append(object_name)
            for unit in unit_name_dict[object_name].next:
                _insert_unit(network, container_node, unit, unit_name_dict, 1)

    elif target_place == 1:
        if object_name not in container_node.affect_ipc:
            container_node.affect_ipc.append(object_name)
            for unit in unit_name_dict[object_name].previous:
                _insert_unit(network, container_node, unit, unit_name_dict, 0)


def network_analysis(network, data):
    graph, _, _ = torch.onnx.utils._model_to_graph(network, data, _retain_param_name=True)

    node_index_link = {}
    unit_index2name = {}
    unit_index_link = {}
    for node in graph.nodes():
        unit_index = node.outputs().__next__().unique()
        node_index_link[unit_index] = []
        node_name = _get_node_name(list(node.inputs())[-1].debugName())
        # only append existing operations
        for input_node in node.inputs():
            input_index = input_node.unique()
            if input_index == 0 or input_index in node_index_link.keys():
                node_index_link[unit_index].append(input_index)
        # backtrace units
        if node.kind() in SUPPORT_LAYER_TUPLE:
            eval("network" + node_name).is_layer = True
            unit_index2name[unit_index] = node_name
            unit_index_link[unit_index] = _backtracking(node_index_link, unit_index, unit_index_link.keys())
        elif node.kind() in SUPPORT_NON_LAYER_TUPLE:
            eval("network" + node_name).is_layer = False
            unit_index2name[unit_index] = node_name
            unit_index_link[unit_index] = _backtracking(node_index_link, unit_index, unit_index_link.keys())

    unit_name_dict = {}
    for unit_index in unit_index_link:
        unit_name_dict[unit_index2name[unit_index]] = LinkNode(unit_index2name[unit_index])
    for unit_index in unit_index_link:
        unit_name = unit_index2name[unit_index]
        # Constructing a doubly-linked-list-like object
        for linked_unit_index in unit_index_link[unit_index]:
            if linked_unit_index != 0:
                linked_unit_name = unit_index2name[linked_unit_index]
                unit_name_dict[unit_name].previous.append(linked_unit_name)
                unit_name_dict[linked_unit_name].next.append(unit_name)

    for unit_name in unit_name_dict:
        unit = unit_name_dict[unit_name]
        _insert_unit(network, unit, unit.name, unit_name_dict, 0)

    return unit_name_dict


if __name__ == "__main__":
    import torchvision.models as models

    model = models.resnet18()
    input_data = torch.ones(1, 3, 224, 224)
    link = network_analysis(model, input_data)
