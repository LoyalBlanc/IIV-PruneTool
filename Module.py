import torch
import torch.nn as nn
import numpy as np


class PruningConvolution(nn.Module):
    def __init__(self, ipc, opc, dilation=1, stride=1, inplace=True):
        super(PruningConvolution, self).__init__()
        self._ipc = ipc
        self._opc = opc
        self._dilation = dilation
        self._stride = stride

        self._conv = None
        self._create_new_conv()
        self._bn = None
        self._create_new_bn()
        self._relu = nn.ReLU(inplace=inplace)

        self._prune_score = None
        self._pruned_channel_index = None
        self.pruned_channel_score = None

    def forward(self, x):
        return self._relu(self._bn(self._conv(x)))

    def _create_new_conv(self, weight=None):
        self._conv = nn.Conv2d(self._ipc, self._opc, 3,
                               stride=self._stride,
                               padding=self._dilation,
                               dilation=self._dilation,
                               bias=False)
        if weight is not None:
            self._conv.weight = nn.Parameter(weight)

    def _create_new_bn(self, bn_weight=None, bn_bias=None):
        self._bn = nn.BatchNorm2d(self._opc)
        if bn_weight is not None:
            self._bn.weight = nn.Parameter(bn_weight)
            self._bn.bias = nn.Parameter(bn_bias)

    def calculate_channel_contribution(self):
        # Calculate contribution based on every opc parameters
        self._prune_score = torch.norm(self._conv.weight[0], p=1).view(1)
        for index in range(1, self._opc):
            self._prune_score = torch.cat((self._prune_score, torch.norm(self._conv.weight[index], p=1).view(1)), dim=0)
        self._prune_score = nn.Softmax(dim=0)(self._prune_score)
        self.pruned_channel_score, self._pruned_channel_index = torch.min(self._prune_score, dim=0)

    def prune_ipc(self, prune_ipc_index):
        # Prune ipc if the conv before has been pruned
        weight = torch.cat((self._conv.weight[:, 0:prune_ipc_index],
                            self._conv.weight[:, prune_ipc_index + 1:]), dim=1)
        self._ipc -= 1
        self._create_new_conv(weight)

    def prune_opc(self, prune_opc_index=None):
        # Prune opc which has the smallest contribution
        # Return the opc_index for the next conv to prune ipc
        if prune_opc_index is None:
            prune_opc_index = self._pruned_channel_index
        weight = torch.cat((self._conv.weight[0:prune_opc_index],
                            self._conv.weight[prune_opc_index + 1:]), dim=0)
        self._opc -= 1
        self._create_new_conv(weight)
        bn_weight = torch.cat((self._bn.weight[0:prune_opc_index], self._bn.weight[prune_opc_index + 1:]), dim=0)
        bn_bias = torch.cat((self._bn.bias[0:prune_opc_index], self._bn.bias[prune_opc_index + 1:]), dim=0)
        self._create_new_bn(bn_weight, bn_bias)
        return prune_opc_index

    def get_pruned_channel(self):
        return self._ipc, self._opc


class PruningDepthConv(nn.Module):
    def __init__(self, ipc, opc, stride=1, dilation=1):
        super(PruningDepthConv, self).__init__()
        self.ipc = ipc
        self.opc = opc
        self.dilation = dilation
        self.stride = stride
        self.squeeze = max(opc // 8, 8)

        self.conv_3 = nn.Conv2d(ipc, ipc, 3, stride=1, padding=dilation, dilation=dilation, groups=ipc, bias=False)
        self.bn_3 = nn.BatchNorm2d(ipc)
        self.conv_1 = nn.Conv2d(ipc, opc, 1, stride=stride, padding=0, dilation=1, groups=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(opc)
        self.fc_2 = nn.Linear(2 * opc, self.squeeze)
        self.fc_1 = nn.Linear(self.squeeze, opc)
        self.relu = nn.ReLU(inplace=True)

        self.pruned_channel_score = 0
        self.pruned_channel_index = 0

    def forward(self, x):
        middle = self.relu(self.bn_3(self.conv_3(x)))
        result = self.relu(self.bn_1(self.conv_1(middle)))
        vector = self.calculate_channel_contribution()
        return result * vector

    def calculate_channel_contribution(self):
        vector_fuse = torch.cat((self.bn_1.weight, self.bn_1.bias), dim=0)
        vector_prune = nn.Softmax(dim=0)(self.fc_1(self.relu(self.fc_2(vector_fuse))))
        self.pruned_channel_score, self.pruned_channel_index = torch.min(vector_prune, dim=0)
        return vector_prune.view(1, self.opc, 1, 1)

    def prune_ipc(self, prune_ipc_index):
        self.ipc = self.ipc - 1
        # conv_3
        weight_conv_3 = torch.cat((self.conv_3.weight[0:prune_ipc_index],
                                   self.conv_3.weight[prune_ipc_index + 1:]), dim=0)
        self.conv_3 = nn.Conv2d(self.ipc, self.ipc, 3, stride=1, padding=self.dilation,
                                dilation=self.dilation, groups=self.ipc, bias=False)
        self.conv_3.weight = nn.Parameter(weight_conv_3)
        # bn_3
        weight_bn_3 = torch.cat((self.bn_3.weight[0:prune_ipc_index], self.bn_3.weight[prune_ipc_index + 1:]), dim=0)
        bias_bn_3 = torch.cat((self.bn_3.bias[0:prune_ipc_index], self.bn_3.bias[prune_ipc_index + 1:]), dim=0)
        self.bn_3 = nn.BatchNorm2d(self.ipc)
        self.bn_3.weight = nn.Parameter(weight_bn_3)
        self.bn_3.bias = nn.Parameter(bias_bn_3)
        # conv_1
        weight_conv_1 = torch.cat((self.conv_1.weight[:, 0:prune_ipc_index],
                                   self.conv_1.weight[:, prune_ipc_index + 1:]), dim=1)
        self.conv_1 = nn.Conv2d(self.ipc, self.opc, 1, stride=self.stride, padding=0,
                                dilation=1, groups=1, bias=False)
        self.conv_1.weight = nn.Parameter(weight_conv_1)

    def prune_opc(self, prune_opc_index=None):
        if prune_opc_index is None:
            prune_opc_index = self.pruned_channel_index
        self.opc = self.opc - 1
        # conv_1
        weight_conv_1 = torch.cat((self.conv_1.weight[0:prune_opc_index],
                                   self.conv_1.weight[prune_opc_index + 1:]), dim=0)
        self.conv_1 = nn.Conv2d(self.ipc, self.opc, 1, stride=self.stride, padding=0,
                                dilation=1, groups=1, bias=False)
        self.conv_1.weight = nn.Parameter(weight_conv_1)
        # bn_1
        weight_bn_1 = torch.cat((self.bn_1.weight[0:prune_opc_index], self.bn_1.weight[prune_opc_index + 1:]), dim=0)
        bias_bn_1 = torch.cat((self.bn_1.bias[0:prune_opc_index], self.bn_1.bias[prune_opc_index + 1:]), dim=0)
        self.bn_1 = nn.BatchNorm2d(self.opc)
        self.bn_1.weight = nn.Parameter(weight_bn_1)
        self.bn_1.bias = nn.Parameter(bias_bn_1)
        # fc_2
        weight_fc_2 = torch.cat((self.fc_2.weight[:, 0:prune_opc_index],
                                 self.fc_2.weight[:, prune_opc_index + 1:self.opc + prune_opc_index],
                                 self.fc_2.weight[:, self.opc + prune_opc_index + 1:]), dim=0)
        bias_fc_2 = self.fc_2.bias
        self.fc_2 = nn.Linear(2 * self.opc, self.squeeze)
        self.fc_2.weight = nn.Parameter(weight_fc_2)
        self.fc_2.bias = nn.Parameter(bias_fc_2)
        # fc_1
        weight_fc_1 = torch.cat((self.fc_1.weight[0:prune_opc_index], self.fc_1.weight[prune_opc_index + 1:]), dim=0)
        bias_fc_1 = torch.cat((self.fc_1.bias[0:prune_opc_index], self.fc_1.bias[prune_opc_index + 1:]), dim=0)
        self.fc_1 = nn.Linear(self.squeeze, self.opc)
        self.fc_1.weight = nn.Parameter(weight_fc_1)
        self.fc_1.bias = nn.Parameter(bias_fc_1)
        return prune_opc_index

    def get_pruned_channel(self):
        return self.ipc, self.opc


if __name__ == "__main__":
    test_data = torch.Tensor(np.ones([7, 8, 11, 1]))
    test_conv = PruningDepthConv(8, 9, dilation=3)
    print(test_conv(test_data).shape)

    test_data = torch.Tensor(np.ones([7, 7, 13, 13]))
    test_conv.prune_ipc(1)
    print(test_conv(test_data).shape)

    test_conv.prune_opc()
    print(test_conv(test_data).shape)

    test_conv.prune_opc(2)
    print(test_conv(test_data).shape)
