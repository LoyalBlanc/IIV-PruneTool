import torch
import torch.nn as nn
from Module.AbstractModule import Abstract


class Basic(Abstract):
    def __init__(self, ipc, opc, stride=1):
        super().__init__(ipc, opc, stride)

    def forward(self, *x):
        feature = self.relu(self.bn(self.conv(x[0])))
        return feature

    def calculate_channel_contribution(self):
        self.score = nn.Softmax(dim=0)(torch.Tensor([torch.norm(weight, p=2) for weight in self.conv.weight]))


if __name__ == "__main__":
    from Module.AbstractModule import module_test

    module_test(Basic, ipc=5, opc=6, data_size=32, stride=2, batch_size=4)
