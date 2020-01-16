from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn


class BasicModule(object):
    __metaclass__ = ABCMeta

    def __init__(self, connect_flag=False):
        self.connect_flag = connect_flag

    @abstractmethod
    def prune_ipc(self, prune_ipc_index):
        pass

    @abstractmethod
    def prune_opc(self, prune_opc_index):
        pass


class Conv2d(nn.Conv2d, BasicModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        BasicModule.__init__(self, connect_flag=False)

    def prune_ipc(self, prune_ipc_index):
        self.in_channels -= 1
        conv_weight = torch.cat((self.weight[:, 0:prune_ipc_index], self.weight[:, prune_ipc_index + 1:]), dim=1)
        self.weight = nn.Parameter(conv_weight)

    def prune_opc(self, prune_opc_index):
        self.out_channels -= 1
        conv_weight = torch.cat((self.weight[0:prune_opc_index], self.weight[prune_opc_index + 1:]), dim=0)
        self.weight = nn.Parameter(conv_weight)
        if self.bias is not None:
            conv_bias = torch.cat((self.bias[0:prune_opc_index], self.bias[prune_opc_index + 1:]), dim=0)
            self.bias = nn.Parameter(conv_bias)


class BatchNorm2d(nn.BatchNorm2d, BasicModule):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False):
        nn.BatchNorm2d.__init__(self, num_features, eps, momentum, affine, track_running_stats)
        BasicModule.__init__(self, connect_flag=True)

    def prune_ipc(self, prune_ipc_index):
        self.num_features -= 1
        bn_weight = torch.cat((self.weight[0:prune_ipc_index], self.weight[prune_ipc_index + 1:]), dim=0)
        bn_bias = torch.cat((self.bias[0:prune_ipc_index], self.bias[prune_ipc_index + 1:]), dim=0)
        self.weight = nn.Parameter(bn_weight)
        self.bias = nn.Parameter(bn_bias)

    def prune_opc(self, prune_opc_index):
        return
