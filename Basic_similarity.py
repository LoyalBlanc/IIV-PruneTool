"""
    if A+B=0: del A,B           Error: Feature changes after ReLU
    if A-B=0: double A del B
"""
import torch
import torch.nn as nn
from Basic import Basic


class BasicSimilarity(Basic):
    def calculate_channel_contribution(self):
        for i in range(self._opc):
            weight = self.conv.weight[i]
            for j in range(i + 1, self._opc):
                temp_score = torch.norm(weight - self.conv.weight[j], p=2)
                if i == 0 and j == 1:
                    score = temp_score
                    self._prune_index = 1
                elif temp_score < score:
                    score = temp_score
                    self._prune_index = j


if __name__ == "__main__":
    from Abstract import prune_test

    basic_similarity = Basic(4, 5, 2)
    prune_test(basic_similarity)
