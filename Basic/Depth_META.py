"""
    Depthwise Separable Convolution with META learning
"""
import torch
import torch.nn as nn
from Depth import DepthwiseSeparable


class DepthwiseSeparableMETA(DepthwiseSeparable):
    def __init__(self, ipc, opc, stride):
        super().__init__(ipc, opc, stride)
        self._squeeze = max(self._opc // 8, 8)
        self.fc_2 = nn.Linear(2 * self._opc, self._squeeze)
        self.fc_1 = nn.Linear(self._squeeze, self._opc)

    def forward(self, x):
        feature = super().forward(x)
        vector = self.calculate_channel_contribution()
        return feature * vector

    def calculate_channel_contribution(self):
        vector_fuse = torch.cat((self.bn_1.weight, self.bn_1.bias), dim=0)
        vector_prune = nn.Softmax(dim=0)(self.fc_1(self.relu(self.fc_2(vector_fuse))))
        _, self._prune_index = torch.min(vector_prune, dim=0)
        return vector_prune.view(1, self.opc, 1, 1)

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
        # fc_2
        weight_fc_2 = torch.cat((self.fc_2.weight[:, 0:prune_opc_index],
                                 self.fc_2.weight[:, prune_opc_index + 1:self.opc + prune_opc_index],
                                 self.fc_2.weight[:, self.opc + prune_opc_index + 1:]), dim=0)
        bias_fc_2 = self.fc_2.bias
        self.fc_2 = nn.Linear(2 * self._opc, self._squeeze)
        self.fc_2.weight = nn.Parameter(weight_fc_2)
        self.fc_2.bias = nn.Parameter(bias_fc_2)
        # fc_1
        weight_fc_1 = torch.cat((self.fc_1.weight[0:prune_opc_index], self.fc_1.weight[prune_opc_index + 1:]), dim=0)
        bias_fc_1 = torch.cat((self.fc_1.bias[0:prune_opc_index], self.fc_1.bias[prune_opc_index + 1:]), dim=0)
        self.fc_1 = nn.Linear(self._squeeze, self._opc)
        self.fc_1.weight = nn.Parameter(weight_fc_1)
        self.fc_1.bias = nn.Parameter(bias_fc_1)
        # Refresh prune score
        self.calculate_channel_contribution()
        return prune_opc_index


if __name__ == "__main__":
    from Basic.Abstract import prune_test

    depth_meta = DepthwiseSeparable(4, 5, 2)
    prune_test(depth_meta)
