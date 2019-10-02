import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class Abstract(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, ipc, opc, stride):
        nn.Module.__init__(self)
        self._ipc = ipc
        self._opc = opc
        self._stride = stride
        self.score = torch.zeros(self.opc)

    @abstractmethod
    def forward(self, *x):
        pass

    def get_channel_contribution(self):
        return self.score

    @abstractmethod
    def prune_ipc(self, pruned_ipc_index):
        self._ipc -= 1

    @abstractmethod
    def prune_opc(self, pruned_opc_index):
        self._opc -= 1
