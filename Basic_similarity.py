"""
    if A+B=0: del A,B           Error: Feature changes after ReLU
    if A-B=0: double A del B
"""
import torch
import torch.nn as nn
from Basic import Basic


class BasicSimilarity(Basic):
    def calculate_channel_contribution(self):
        self._prune_index = [0, 1]
        score_weight = torch.norm(self.conv.weight[0], p=2)
        score_similarity = torch.norm(self.conv.weight[0] - self.conv.weight[1], p=2)
        for i in range(self._opc):
            weight = self.conv.weight[i]
            if i != 0:
                temp_score_weight = torch.norm(weight, p=2)
                if score_weight > temp_score_weight:
                    score_weight = temp_score_weight
                    self._prune_index[0] = i
            for j in range(i + 1, self._opc):
                if not (i == 0 and j == 1):
                    temp_score_similarity = torch.norm(weight - self.conv.weight[j], p=2)
                    if temp_score_similarity < score_similarity:
                        score_similarity = temp_score_similarity
                        self._prune_index[1] = j
        # print(self._prune_index)

    def prune_opc(self, prune_opc_index=None):
        if prune_opc_index is None:
            prune_opc_index = self._prune_index
        if prune_opc_index[0] == prune_opc_index[1]:
            super().prune_opc(prune_opc_index[0])
        else:
            self._opc -= 2
            prune_opc_index_1 = min(prune_opc_index)
            prune_opc_index_2 = max(prune_opc_index)
            conv_weight = torch.cat((self.conv.weight[0:prune_opc_index_1],
                                     self.conv.weight[prune_opc_index_1 + 1:prune_opc_index_2],
                                     self.conv.weight[prune_opc_index_2 + 1:]), dim=0)
            self.conv = nn.Conv2d(self._ipc, self._opc, 3, stride=self._stride, padding=1, bias=False)
            self.conv.weight = nn.Parameter(conv_weight)

            bn_weight = torch.cat((self.bn.weight[0:prune_opc_index_1],
                                   self.bn.weight[prune_opc_index_1 + 1:prune_opc_index_2],
                                   self.bn.weight[prune_opc_index_2 + 1:]), dim=0)
            bn_bias = torch.cat((self.bn.bias[0:prune_opc_index_1],
                                 self.bn.bias[prune_opc_index_1 + 1:prune_opc_index_2],
                                 self.bn.bias[prune_opc_index_2 + 1:]), dim=0)
            self.bn = nn.BatchNorm2d(self._opc)
            self.bn.weight = nn.Parameter(bn_weight)
            self.bn.bias = nn.Parameter(bn_bias)
        self.calculate_channel_contribution()
        return prune_opc_index


if __name__ == "__main__":
    from Abstract import prune_test

    basic_similarity = BasicSimilarity(4, 10, 2)
    prune_test(basic_similarity)
