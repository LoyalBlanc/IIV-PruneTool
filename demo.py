import os

import torch
import torch.nn as nn

import modules.abstract_network as an
import modules.basic_module as bm


class DemoNet(an.AbstractNetwork):
    def __init__(self):
        an.AbstractNetwork.__init__(self)

        self.conv1 = bm.Conv2d(5, 4, 3, padding=1)
        self.conv2 = bm.Conv2d(4, 4, 3, padding=1)
        self.conv3 = bm.Conv2d(4, 5, 3, padding=1)
        self.conv4 = bm.Conv2d(5, 4, 3, padding=1)
        self.conv5 = bm.Conv2d(4, 4, 3, padding=1)

        self.bn1 = bm.BatchNorm2d(4)
        self.bn2 = bm.BatchNorm2d(4)
        self.bn3 = bm.BatchNorm2d(5)
        self.bn4 = bm.BatchNorm2d(4)
        self.bn5 = bm.BatchNorm2d(4)

        self.activate = nn.ReLU()

        self.network_analysis(5)

    def forward(self, input_tensor):
        x1 = self.activate(self.bn1(self.conv1(input_tensor)))
        x2 = self.activate(self.bn2(self.conv2(x1)))
        x3 = self.activate(self.bn3(self.conv3(x2 + x1)))
        x4 = self.activate(self.bn4(self.conv4(x3 + input_tensor)))
        x5 = self.activate(self.bn5(self.conv5(x4 + x2 + x1)))
        return x5


if __name__ == "__main__":
    torch.manual_seed(229)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    net = DemoNet()
