import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable

import Module.basic_module as bm
import Module.abstract_network as an

import PruningMethod.methods as pm
import mnist


class DemoNet(an.AbstractNetwork):
    def __init__(self):
        an.AbstractNetwork.__init__(self)
        self.layer_trunk = nn.Sequential(bm.BasicModule(1, 16, 3), bm.BasicModule(16, 16, 3),
                                         bm.BasicModule(16, 16, 3), bm.Conv2d(16, 10, 3, padding=1))

        self.link_matrix = np.array([[0, 0, 0, 0],
                                     [1, 0, 0, 0],
                                     [1, 1, 0, 0],
                                     [1, 1, 1, 0]])

    def forward(self, x):
        x = an.AbstractNetwork.forward(self, x)
        return nn.AdaptiveMaxPool2d(1)(x).squeeze_()


if __name__ == "__main__":
    torch.manual_seed(229)

    demo_data = Variable(torch.rand(1, 1, 32, 32), requires_grad=True)
    demo_net = DemoNet()
    mnist.train_model(demo_net)
    # dataset.valid_model(demo_net)

    # pm.one_shot_prune(demo_net, demo_data, method="minimum_weight", prune_ratio=0.2)
