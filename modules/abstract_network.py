from abc import ABCMeta, abstractmethod

import torch.nn as nn

import utils.utils_torch as ut


class LinkNode(object):
    def __init__(self, name):
        self.name = name
        self.up = []
        self.down = []
        self.opc = []
        self.ipc = []

    def get_info(self):
        basic_info = "Node " + self.name + ': [' + ",".join(self.up) + "] -> " \
                     + self.name + "(This Node) -> [" + ",".join(self.down) \
                     + "]\nPrune influence:\n\topc: [" + ",".join(self.opc) \
                     + "]\n\tipc: [" + ",".join(self.ipc) + "]\n"
        return basic_info


class AbstractNetwork(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        nn.Module.__init__(self)
        self.layer_name_link = {}

    @abstractmethod
    def forward(self, input_tensor):
        pass

    def network_analysis(self, input_channel):
        node_index2name, layer_index_link = ut.network_link_analysis(self, input_channel)
        for node_index in layer_index_link:
            self.layer_name_link[node_index2name[node_index]] = LinkNode(node_index2name[node_index])
        for node_index in layer_index_link:
            node_name = node_index2name[node_index]
            for item in layer_index_link[node_index]:
                if item != 0:
                    item_name = node_index2name[item]
                    self.layer_name_link[node_name].up.append(item_name)
                    self.layer_name_link[item_name].down.append(node_name)
        for node_name in self.layer_name_link:
            node = self.layer_name_link[node_name]
            self.append_new_node(node, node.name, 0)

    def append_new_node(self, container_node, object_name, target_place=0):
        connect_flag = eval("self." + object_name).connect_flag
        if connect_flag or target_place == 0:
            if object_name not in container_node.opc:
                container_node.opc.append(object_name)
                for item in self.layer_name_link[object_name].down:
                    self.append_new_node(container_node, item, 1)

        if connect_flag or target_place == 1:
            if object_name not in container_node.ipc:
                container_node.ipc.append(object_name)
                for item in self.layer_name_link[object_name].up:
                    self.append_new_node(container_node, item, 0)

    def __repr__(self):
        info = ''
        for item in self.layer_name_link:
            info += self.layer_name_link[item].get_info()
        return info
