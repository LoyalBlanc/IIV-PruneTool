from abc import ABCMeta, abstractmethod

import torch.nn as nn


class AbstractNetwork(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        nn.Module.__init__(self)

    @abstractmethod
    def forward(self, input_tensor):
        pass
