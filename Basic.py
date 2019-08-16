"""
    Basic convolution module for pruning
"""
import torch
import torch.nn as nn
from Abstract import Abstract


class Basic(Abstract):
    def __init__(self, ipc, opc, stride):
        super().__init__(ipc, opc, stride)
        self.conv = nn.Conv2d(self._ipc, self._opc, 3, stride=self._stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self._opc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *x):
        feature = self.relu(self.bn(self.conv(x[0])))
        return [feature, *x[1:]]

    def calculate_channel_contribution(self):
        self._prune_score = torch.Tensor([torch.norm(weight, p=2) for weight in self.conv.weight])
        _, self._prune_index = torch.min(nn.Softmax(dim=0)(self._prune_score), dim=0)

    def prune_ipc(self, prune_ipc_index):
        super().prune_ipc(prune_ipc_index)
        conv_weight = torch.cat((self.conv.weight[:, 0:prune_ipc_index],
                                 self.conv.weight[:, prune_ipc_index + 1:]), dim=1)
        self.conv = nn.Conv2d(self._ipc, self._opc, 3, stride=self._stride, padding=1, bias=False)
        self.conv.weight = nn.Parameter(conv_weight)

    def prune_opc(self, prune_opc_index=None):
        prune_opc_index = super().prune_opc(prune_opc_index)

        conv_weight = torch.cat((self.conv.weight[0:prune_opc_index], self.conv.weight[prune_opc_index + 1:]), dim=0)
        self.conv = nn.Conv2d(self._ipc, self._opc, 3, stride=self._stride, padding=1, bias=False)
        self.conv.weight = nn.Parameter(conv_weight)

        bn_weight = torch.cat((self.bn.weight[0:prune_opc_index], self.bn.weight[prune_opc_index + 1:]), dim=0)
        bn_bias = torch.cat((self.bn.bias[0:prune_opc_index], self.bn.bias[prune_opc_index + 1:]), dim=0)
        self.bn = nn.BatchNorm2d(self._opc)
        self.bn.weight = nn.Parameter(bn_weight)
        self.bn.bias = nn.Parameter(bn_bias)

        self.calculate_channel_contribution()
        return prune_opc_index

    def get_pruned_parameter(self):
        return self.conv.weight, self.bn.weight, self.bn.bias


if __name__ == "__main__":
    from Abstract import prune_test

    basic = Basic(4, 5, 2)
    prune_test(basic)
