"""
    Depthwise Separable Convolution
"""
import torch
import torch.nn as nn
from Abstract import Abstract


class DepthwiseSeparable(Abstract):
    def __init__(self, ipc, opc, stride):
        super().__init__(ipc, opc, stride)
        self._conv_3 = nn.Conv2d(self._ipc, self._ipc, 3, stride=self._stride, padding=1, groups=self._ipc, bias=False)
        self._bn_3 = nn.BatchNorm2d(self._ipc)
        self._conv_1 = nn.Conv2d(self._ipc, self._opc, 1, bias=False)
        self._bn_1 = nn.BatchNorm2d(self._opc)
        self._relu = nn.ReLU(inplace=True)

    def forward(self, *x):
        feature = self._relu(self._bn_3(self._conv_3(x[0])))
        feature = self._relu(self._bn_1(self._conv_1(feature)))
        return [feature, *x[1:]]

    def calculate_channel_contribution(self):
        # Calculate contribution based on every opc parameters
        # Only conv_1 will be considered
        prune_score = torch.Tensor([torch.norm(weight, p=2) for weight in self._conv_1.weight])
        _, self._prune_index = torch.min(nn.Softmax(dim=0)(prune_score), dim=0)

    def prune_ipc(self, prune_ipc_index):
        super().prune_ipc(prune_ipc_index)
        # conv_3
        conv_3_weight = torch.cat((self._conv_3.weight[0:prune_ipc_index],
                                   self._conv_3.weight[prune_ipc_index + 1:]), dim=0)
        self._conv_3 = nn.Conv2d(self._ipc, self._ipc, 3, stride=self._stride, padding=1, groups=self._ipc, bias=False)
        self._conv_3.weight = nn.Parameter(conv_3_weight)
        # bn_3
        bn_3_weight = torch.cat((self._bn_3.weight[0:prune_ipc_index], self._bn_3.weight[prune_ipc_index + 1:]), dim=0)
        bn_3_bias = torch.cat((self._bn_3.bias[0:prune_ipc_index], self._bn_3.bias[prune_ipc_index + 1:]), dim=0)
        self._bn_3 = nn.BatchNorm2d(self._ipc)
        self._bn_3.weight = nn.Parameter(bn_3_weight)
        self._bn_3.bias = nn.Parameter(bn_3_bias)
        # conv_1
        conv_1_weight = torch.cat((self._conv_1.weight[:, 0:prune_ipc_index],
                                   self._conv_1.weight[:, prune_ipc_index + 1:]), dim=1)
        self._conv_1 = nn.Conv2d(self._ipc, self._opc, 1, bias=False)
        self._conv_1.weight = nn.Parameter(conv_1_weight)

    def prune_opc(self, prune_opc_index=None):
        prune_opc_index = super().prune_opc(prune_opc_index)
        # conv_1
        conv_1_weight = torch.cat((self._conv_1.weight[0:prune_opc_index],
                                   self._conv_1.weight[prune_opc_index + 1:]), dim=0)
        self._conv_1 = nn.Conv2d(self._ipc, self._opc, 1, bias=False)
        self._conv_1.weight = nn.Parameter(conv_1_weight)
        # bn_1
        bn_1_weight = torch.cat((self._bn_1.weight[0:prune_opc_index], self._bn_1.weight[prune_opc_index + 1:]), dim=0)
        bn_1_bias = torch.cat((self._bn_1.bias[0:prune_opc_index], self._bn_1.bias[prune_opc_index + 1:]), dim=0)
        self._bn_1 = nn.BatchNorm2d(self._opc)
        self._bn_1.weight = nn.Parameter(bn_1_weight)
        self._bn_1.bias = nn.Parameter(bn_1_bias)
        # Refresh prune score
        self.calculate_channel_contribution()
        return prune_opc_index

    def get_pruned_parameter(self):
        return [self._conv_3.weight, self._bn_3.weight, self._bn_3.bias,
                self._conv_1.weight, self._bn_1.weight, self._bn_1.bias]


if __name__ == "__main__":
    from Abstract import prune_test

    depth = DepthwiseSeparable(4, 5, 2)
    prune_test(depth)
