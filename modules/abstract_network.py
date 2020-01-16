from abc import ABCMeta, abstractmethod

import torch.nn as nn

import utils.utils_torch as utils_torch


class LinkNode(object):
    def __init__(self, name):
        """
            Just like a doubly linked list.
            up: previous node(s)
            down: next node(s)
            opc: node(s) pruning opc with this node at the same time
            ipc: node(s) pruning ipc with this node at the same time
        """
        self.name = name
        self.up = []
        self.down = []
        self.opc = []
        self.ipc = []

    def get_repr_info(self):
        repr_info = "Node " + self.name + ': [' + ",".join(self.up) + "] -> " \
                    + self.name + "(This Node) -> [" + ",".join(self.down) \
                    + "]\nPrune influence:\n\topc: [" + ",".join(self.opc) \
                    + "]\n\tipc: [" + ",".join(self.ipc) + "]\n"
        return repr_info


class AbstractNetwork(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        """
            _layer_name_link: automatic created by self.network_analysis(input_channel)
            pruning_range: decide which layer will be involved in pruning
            hook: some useful functions, created by pruning method
            regularization: the regularization value, calculated by hook
        """
        nn.Module.__init__(self)
        self._layer_name_link = {}
        self.pruning_range = []
        self.hook = None
        self.regularization = None

    @abstractmethod
    def forward(self, input_tensor):
        pass

    def network_analysis(self, input_channel):
        node_index2name, layer_index_link = utils_torch.network_link_analysis(self, input_channel)
        for node_index in layer_index_link:
            self._layer_name_link[node_index2name[node_index]] = LinkNode(node_index2name[node_index])
        for node_index in layer_index_link:
            node_name = node_index2name[node_index]
            for item in layer_index_link[node_index]:
                if item != 0:
                    item_name = node_index2name[item]
                    self._layer_name_link[node_name].up.append(item_name)
                    self._layer_name_link[item_name].down.append(node_name)
        for node_name in self._layer_name_link:
            node = self._layer_name_link[node_name]
            self._insert_node(node, node.name, 0)

    def _insert_node(self, container_node, object_name, target_place=0):
        connect_flag = eval("self." + object_name).connect_flag
        if connect_flag:
            for node in self._layer_name_link[object_name].down:
                self._insert_node(container_node, node, 1)
            for node in self._layer_name_link[object_name].up:
                self._insert_node(container_node, node, 0)

        elif target_place == 0:
            if object_name not in container_node.opc:
                container_node.opc.append(object_name)
                for node in self._layer_name_link[object_name].down:
                    self._insert_node(container_node, node, 1)

        elif target_place == 1:
            if object_name not in container_node.ipc:
                container_node.ipc.append(object_name)
                for node in self._layer_name_link[object_name].up:
                    self._insert_node(container_node, node, 0)

    def prune_spec_channel(self, module, channel):
        layer_info = 'Layer '
        spec_layer = self._layer_name_link[module]
        layer_prune_opc, layer_prune_ipc = spec_layer.opc, spec_layer.ipc
        for layer in layer_prune_opc:
            layer_info += "%s, " % layer
            eval('self.' + layer).prune_opc(channel)
            for sub_layer in self._layer_name_link[layer].down:
                eval('self.' + sub_layer).prune_ipc(channel)
        for layer in layer_prune_ipc:
            eval('self.' + layer).prune_ipc(channel)
        return layer_info

    def get_layer_name_link(self):
        return self._layer_name_link

    def list_modules(self):
        return self._modules

    def __repr__(self):
        repr_info = nn.Module.__repr__(self) + '\n'
        for item in self._layer_name_link:
            repr_info += self._layer_name_link[item].get_repr_info()
        return repr_info
