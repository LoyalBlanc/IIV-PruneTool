import torch.nn as nn
from abc import ABCMeta, abstractmethod


class Abstract(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, ipc, opc, stride):
        super(Abstract, self).__init__()
        self._ipc = ipc
        self._opc = opc
        self._stride = stride

        self._prune_score = None
        self.prune_index = None

    @abstractmethod
    def forward(self, *x):
        pass

    def get_pruned_channel(self):
        return self._ipc, self._opc

    @abstractmethod
    def calculate_channel_contribution(self):
        pass

    @abstractmethod
    def prune_ipc(self, prune_ipc_index):
        pass

    @abstractmethod
    def prune_opc(self, prune_opc_index=None):
        pass
