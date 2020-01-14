import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from Module.basic_module import BasicModule, Conv2d
from Module.abstract_network import AbstractNetwork
from PruneMethod.method import one_shot_prune


class DemoNet(AbstractNetwork):
    def __init__(self):
        AbstractNetwork.__init__(self)
        self.layer_trunk = nn.Sequential(BasicModule(3, 16, 3), BasicModule(16, 16, 3),
                                         BasicModule(16, 16, 3), Conv2d(16, 10, 3, padding=1))

        self.link_matrix = np.tril(np.array([[0, 0, 0, 0],
                                             [1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [1, 0, 1, 0]]), k=-1)
        self.link_matrix_analysis()


if __name__ == "__main__":
    demo_data = Variable(torch.rand(3, 32, 32).unsqueeze(0), requires_grad=True)
    demo_net = DemoNet()
    one_shot_prune(demo_net, demo_data, method="minimum_weight", prune_ratio=0.4)
