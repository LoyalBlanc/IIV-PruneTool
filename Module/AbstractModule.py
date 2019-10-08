import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class Abstract(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, ipc, opc, stride):
        nn.Module.__init__(self)
        self.ipc = ipc
        self.opc = opc
        self.stride = stride

        self.conv = nn.Conv2d(self.ipc, self.opc, 3, stride=self.stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.opc)
        self.relu = nn.ReLU(inplace=True)

        self.score = torch.zeros(self.opc)
        self.regularization = 0

    @abstractmethod
    def forward(self, *x):
        pass

    @abstractmethod
    def calculate_channel_contribution(self):
        pass

    def get_channel_contribution(self):
        return self.score

    def prune_ipc(self, prune_ipc_index):
        self.ipc -= 1
        conv_weight = torch.cat((self.conv.weight[:, 0:prune_ipc_index],
                                 self.conv.weight[:, prune_ipc_index + 1:]), dim=1)
        self.conv = nn.Conv2d(self.ipc, self.opc, 3, stride=self.stride, padding=1, bias=False)
        self.conv.weight = nn.Parameter(conv_weight)

    def prune_opc(self, prune_opc_index):
        self.opc -= 1
        conv_weight = torch.cat((self.conv.weight[0:prune_opc_index], self.conv.weight[prune_opc_index + 1:]), dim=0)
        self.conv = nn.Conv2d(self.ipc, self.opc, 3, stride=self.stride, padding=1, bias=False)
        self.conv.weight = nn.Parameter(conv_weight)

        bn_weight = torch.cat((self.bn.weight[0:prune_opc_index], self.bn.weight[prune_opc_index + 1:]), dim=0)
        bn_bias = torch.cat((self.bn.bias[0:prune_opc_index], self.bn.bias[prune_opc_index + 1:]), dim=0)
        self.bn = nn.BatchNorm2d(self.opc)
        self.bn.weight = nn.Parameter(bn_weight)
        self.bn.bias = nn.Parameter(bn_bias)

        self.calculate_channel_contribution()
        return prune_opc_index


def module_test(module, batch_size=1, ipc=2, opc=3, data_size=4, stride=1):
    test_conv = module(ipc=ipc, opc=opc, stride=stride)
    test_data = torch.randn(batch_size, ipc, data_size, data_size)
    test_output = test_conv(test_data)
    test_conv.calculate_channel_contribution()
    test_score = test_conv.get_channel_contribution()
    print("Origin score:", test_score)
    assert test_score.shape == torch.Size([opc])
    assert test_output.shape == torch.Size([batch_size, opc, data_size // stride, data_size // stride])

    test_conv.prune_ipc(ipc // 2)
    test_data = torch.randn(batch_size, ipc - 1, data_size, data_size)
    test_output = test_conv(test_data)
    test_conv.calculate_channel_contribution()
    test_score = test_conv.get_channel_contribution()
    print("Prune ipc score:", test_score)
    assert test_score.shape == torch.Size([opc])
    assert test_output.shape == torch.Size([batch_size, opc, data_size // stride, data_size // stride])

    test_conv.prune_opc(opc // 2)
    test_output = test_conv(test_data)
    test_conv.calculate_channel_contribution()
    test_score = test_conv.get_channel_contribution()
    print("Prune opc score:", test_score)
    assert test_score.shape == torch.Size([opc - 1])
    assert test_output.shape == torch.Size([batch_size, opc - 1, data_size // stride, data_size // stride])
