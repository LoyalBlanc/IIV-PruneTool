import os

import numpy as np
import old_version.Module.abstract_network as an
import old_version.Module.basic_module as bm
import old_version.PruningMethod.methods as pm
import torch
import torch.nn as nn
from old_version.utils import mnist


class DemoNet(an.AbstractNetwork):
    def __init__(self):
        an.AbstractNetwork.__init__(self)
        self.layer_trunk = nn.Sequential(bm.BasicModule(1, 32, 3),
                                         bm.BasicModule(32, 64, 3),
                                         bm.BasicModule(64, 64, 3),
                                         bm.BasicModule(64, 32, 3),
                                         bm.Conv2d(32, 10, 3, padding=1))

        self.link_matrix = np.array([[0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0],
                                     [0, 1, 1, 0, 0],
                                     [1, 0, 0, 1, 0]])

    def forward(self, x):
        x = an.AbstractNetwork.forward(self, x)
        return nn.AdaptiveMaxPool2d(1)(x).squeeze_()


if __name__ == "__main__":
    torch.manual_seed(229)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    '''demo_flag
        0: Prepare the demo network
        1: One shot pruning
        2: Iterative pruning
        3: Automotive pruning 
    '''

    demo_flag = 3

    if demo_flag == 0:
        demo_net = DemoNet()
        demo_net.before_training()
        mnist.train_model(demo_net, epochs=100, batch_size=3000, regular=True)
        demo_net.after_training()
        mnist.valid_model(demo_net, batch_size=2500)  # 94.97 / FLOPs 600064
        mnist.save_param(demo_net, "demo_param.pkl")
    elif demo_flag == 1:
        demo_net = DemoNet()
        mnist.load_param(demo_net, "demo_param.pkl")
        pm.one_shot_prune(demo_net,
                          method="minimum_weight",
                          prune_ratio=0.1)
        mnist.valid_model(demo_net, batch_size=2500)  # 92.02 / FLOPs 538624 / without training
    elif demo_flag == 2:
        demo_net = DemoNet()
        mnist.load_param(demo_net, "demo_param.pkl")
        pm.iterative_prune(demo_net,
                           mnist.get_train_loader(2500),
                           method="minimum_weight",
                           prune_ratio=0.1,
                           criterion=nn.CrossEntropyLoss())
        mnist.valid_model(demo_net, batch_size=2500)  # 94.90 / FLOPs 538624 / training 10 epochs
    elif demo_flag == 3:
        demo_net = DemoNet()
        mnist.load_param(demo_net, "demo_param.pkl")
        pm.automotive_prune(demo_net,
                            mnist.get_train_loader(2500),
                            method="minimum_weight",
                            prune_ratio=0.4,
                            criterion=nn.CrossEntropyLoss(),
                            epoch_limit=2)
        mnist.valid_model(demo_net, batch_size=2500)  # 90.66 / FLOPs 587776 / training 5 epochs
