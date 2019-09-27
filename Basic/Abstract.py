import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class Abstract(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, ipc, opc, stride):
        super(Abstract, self).__init__()
        self._ipc = ipc
        self._opc = opc
        self._stride = stride

        self._prune_index = None

    @abstractmethod
    def forward(self, *x):
        pass

    @abstractmethod
    def calculate_channel_contribution(self):
        pass

    # def get_prune_score_index(self):
    #     return self._prune_score[self._prune_index].item()

    @abstractmethod
    def prune_ipc(self, prune_ipc_index):
        self._ipc -= 1

    @abstractmethod
    def prune_opc(self, prune_opc_index=None):
        if prune_opc_index is None:
            prune_opc_index = self._prune_index
        self._opc -= 1
        return prune_opc_index

    def get_pruned_channel(self):
        return self._ipc, self._opc

    @abstractmethod
    def get_pruned_parameter(self):
        pass


def prune_test(conv):
    conv.calculate_channel_contribution()
    data = torch.randn(6, conv.get_pruned_channel()[0], 6, 6)
    print("Origin Shape:  {} -> {}".format(data.shape, conv(data)[0].shape))

    conv.prune_opc()
    print("1st Prune Opc: {} -> {}".format(data.shape, conv(data)[0].shape))
    conv.prune_opc()
    print("2nd Prune Opc: {} -> {}".format(data.shape, conv(data)[0].shape))
    conv.prune_opc()
    print("3rd Prune Opc: {} -> {}".format(data.shape, conv(data)[0].shape))

    conv.prune_ipc(1)
    data = torch.randn(6, conv.get_pruned_channel()[0], 6, 6)
    print("1st Prune Ipc: {} -> {}".format(data.shape, conv(data)[0].shape))
    conv.prune_ipc(1)
    data = torch.randn(6, conv.get_pruned_channel()[0], 6, 6)
    print("2nd Prune Ipc: {} -> {}".format(data.shape, conv(data)[0].shape))
    conv.prune_ipc(1)
    data = torch.randn(6, conv.get_pruned_channel()[0], 6, 6)
    print("3rd Prune Ipc: {} -> {}".format(data.shape, conv(data)[0].shape))
