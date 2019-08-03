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


if __name__ == "__main__":
    test_data = torch.Tensor(np.ones([1, 1, 33, 33]))
    test_conv = PruningConvolution(2, 2, dilation=3)
