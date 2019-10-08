import torch
import torch.nn as nn
from Module.AbstractModule import AbstractModule


class Basic(AbstractModule):
    def __init__(self, ipc, opc, stride=1):
        super().__init__(ipc, opc, stride)

    def forward(self, *x):
        feature = self.relu(self.bn(self.conv(x[0])))
        return feature

    def calculate_channel_contribution(self):
        self.score = nn.Softmax(dim=0)(torch.Tensor([torch.norm(weight, p=2) for weight in self.conv.weight]))

    @staticmethod
    def calculate_regularization(module, input, output):
        module.regularization += torch.norm(module.conv.weight, p=1)

    def before_pruning(self):
        self.regularization = 0
        self.hook = self.register_forward_hook(self.calculate_regularization)

    def after_pruning(self):
        self.hook.remove()


if __name__ == "__main__":
    from Module.AbstractModule import module_test

    module_test(Basic, ipc=2, opc=3, data_size=32, stride=2, batch_size=4)
