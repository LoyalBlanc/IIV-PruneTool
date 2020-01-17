import torch
import torch.nn as nn


def layer_prune_ipc(self, prune_ipc_index):
    self.in_channels -= 1
    conv_weight = torch.cat((self.weight[:, 0:prune_ipc_index], self.weight[:, prune_ipc_index + 1:]), dim=1)
    self.weight = nn.Parameter(conv_weight, requires_grad=True)


def layer_prune_opc(self, prune_opc_index):
    self.out_channels -= 1
    conv_weight = torch.cat((self.weight[0:prune_opc_index], self.weight[prune_opc_index + 1:]), dim=0)
    self.weight = nn.Parameter(conv_weight, requires_grad=True)
    if self.bias is not None:
        conv_bias = torch.cat((self.bias[0:prune_opc_index], self.bias[prune_opc_index + 1:]), dim=0)
        self.bias = nn.Parameter(conv_bias, requires_grad=True)
