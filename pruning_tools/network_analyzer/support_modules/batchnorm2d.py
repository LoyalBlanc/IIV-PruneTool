import torch
import torch.nn as nn


def non_layer_prune(self, prune_ipc_index):
    self.num_features -= 1
    bn_weight = torch.cat((self.weight[0:prune_ipc_index], self.weight[prune_ipc_index + 1:]), dim=0)
    bn_bias = torch.cat((self.bias[0:prune_ipc_index], self.bias[prune_ipc_index + 1:]), dim=0)
    self.weight = nn.Parameter(bn_weight, requires_grad=True)
    self.bias = nn.Parameter(bn_bias, requires_grad=True)
