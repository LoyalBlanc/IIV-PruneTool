import torch
import torch.nn as nn
from Module.AbstractModule import AbstractModule


class MinimumWeight(AbstractModule):
    def __init__(self, ipc, opc, stride=1):
        super().__init__(ipc, opc, stride)

    def before_pruning_module(self):
        def calculate_regularization(module, input_tensor, output_tensor):
            module.regularization = torch.norm(module.conv.weight, p=1)

        self.hook = self.register_forward_hook(calculate_regularization)

    def calculate_channel_contribution(self):
        self.score = nn.Softmax(dim=0)(torch.Tensor([torch.norm(weight, p=2) for weight in self.conv.weight]))


if __name__ == "__main__":
    from Module.AbstractModule import module_test

    module_test(MinimumWeight, ipc=2, opc=3, data_size=32, stride=2, batch_size=4)
