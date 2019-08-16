"""
    Depthwise Separable convolution
"""
import torch
import torch.nn as nn


class DepthwiseSeparable(nn.Module):
    def __init__(self, ipc, opc, stride, param, only_conv):
        super(DepthwiseSeparable, self).__init__()
        self._ipc = ipc
        self._opc = opc
        self._stride = stride
        self._dilation = param[0]
        self._only_conv = only_conv
        # Depthwise + Pointwise
        self._relu = nn.ReLU(inplace=True)
        self._conv_3 = None
        self._bn_3 = None
        self._create_new_conv_3(*param[1:4])
        self._conv_1 = None
        self._bn_1 = None
        self._create_new_conv_1(*param[4:])
        # Prune parameters
        self._prune_score = None
        self._pruned_channel_index = None
        self.pruned_channel_score = None

    def forward(self, *x):
        feature = self._conv_1(self._relu(self._bn_3(self._conv_3(x[0]))))
        if not self._only_conv:
            feature = self._relu(self._bn_1(feature))
        return feature, x[1]

    def _create_new_conv_3(self, *param, with_bn=True):
        # Rebuild conv_3
        # param: conv_weight, bn_weight, bn_bias
        self._conv_3 = nn.Conv2d(self._ipc, self._ipc, 3, stride=self._stride, padding=self._dilation,
                                 dilation=self._dilation, groups=self._ipc, bias=False)
        if param[0] is not None:
            self._conv_3.weight = nn.Parameter(param[0])

        if with_bn:
            self._bn_3 = nn.BatchNorm2d(self._ipc)
            if param[1] is not None:
                self._bn_3.weight = nn.Parameter(param[1])
                self._bn_3.bias = nn.Parameter(param[2])

    def _create_new_conv_1(self, *param, with_bn=True):
        # Rebuild conv_1
        # param: conv_weight, bn_weight, bn_bias
        self._conv_1 = nn.Conv2d(self._ipc, self._opc, 1, bias=False)
        if param[0] is not None:
            self._conv_1.weight = nn.Parameter(param[0])

        if with_bn and not self._only_conv:
            self._bn_1 = nn.BatchNorm2d(self._opc)
            if param[1] is not None:
                self._bn_1.weight = nn.Parameter(param[1])
                self._bn_1.bias = nn.Parameter(param[2])

    def get_pruned_channel(self):
        return self._ipc, self._opc

    def get_pruned_parameter(self):
        return [self._conv_3.weight, self._bn_3.weight, self._bn_3.bias,
                self._conv_1.weight, self._bn_1.weight, self._bn_1.bias]

    def calculate_channel_contribution(self):
        # Calculate contribution based on every opc parameters
        # Only conv_1 will be considered
        self._prune_score = torch.cat(([torch.norm(weight, p=2).view(1) for weight in self._conv_1.weight]), dim=0)
        self.pruned_channel_score, self._pruned_channel_index = torch.min(nn.Softmax(dim=0)(self._prune_score), dim=0)

    def prune_ipc(self, prune_ipc_index):
        # Prune ipc if the conv before has been pruned
        self._ipc = self._ipc - 1
        # conv_3 & bn_3
        weight_conv_3 = torch.cat((self._conv_3.weight[0:prune_ipc_index],
                                   self._conv_3.weight[prune_ipc_index + 1:]), dim=0)
        weight_bn_3 = torch.cat((self._bn_3.weight[0:prune_ipc_index], self._bn_3.weight[prune_ipc_index + 1:]), dim=0)
        bias_bn_3 = torch.cat((self._bn_3.bias[0:prune_ipc_index], self._bn_3.bias[prune_ipc_index + 1:]), dim=0)
        self._create_new_conv_3(weight_conv_3, weight_bn_3, bias_bn_3)
        # conv_1
        weight_conv_1 = torch.cat((self._conv_1.weight[:, 0:prune_ipc_index],
                                   self._conv_1.weight[:, prune_ipc_index + 1:]), dim=1)
        self._create_new_conv_1(weight_conv_1, with_bn=False)

    def prune_opc(self, prune_opc_index=None):
        # Prune opc will only change conv_1
        if prune_opc_index is None:
            prune_opc_index = self._pruned_channel_index
        self._opc = self._opc - 1
        # conv_1
        weight_conv_1 = torch.cat((self._conv_1.weight[0:prune_opc_index],
                                   self._conv_1.weight[prune_opc_index + 1:]), dim=0)
        # bn_1
        weight_bn_1 = torch.cat((self._bn_1.weight[0:prune_opc_index], self._bn_1.weight[prune_opc_index + 1:]), dim=0)
        bias_bn_1 = torch.cat((self._bn_1.bias[0:prune_opc_index], self._bn_1.bias[prune_opc_index + 1:]), dim=0)
        self._create_new_conv_1(weight_conv_1, weight_bn_1, bias_bn_1)
        # Refresh prune score
        self.calculate_channel_contribution()
        return prune_opc_index
