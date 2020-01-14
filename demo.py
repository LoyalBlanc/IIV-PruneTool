import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import Module.abstract_network as an
import Module.basic_module as bm
import PruningMethod.methods as pm
import mnist


class DemoNet(an.AbstractNetwork):
    def __init__(self):
        an.AbstractNetwork.__init__(self)
        self.layer_trunk = nn.Sequential(bm.BasicModule(1, 32, 3), bm.BasicModule(32, 64, 3),
                                         bm.BasicModule(64, 64, 3), bm.BasicModule(64, 32, 3),
                                         bm.Conv2d(32, 10, 3, padding=1))

        self.link_matrix = np.array([[0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0],
                                     [0, 1, 1, 0, 0],
                                     [1, 0, 0, 1, 0]])
        self.hook = None
        self.regularization = None

    def forward(self, x):
        x = an.AbstractNetwork.forward(self, x)
        return nn.AdaptiveMaxPool2d(1)(x).squeeze_()

    def before_training(self):
        def conv_hook(network, input_tensor, output_tensor):
            for module in self.layer_trunk:
                if isinstance(module, bm.BasicModule):
                    network.regularization += torch.norm(module.conv.weight, p=1)
                if isinstance(module, bm.Conv2d):
                    network.regularization += torch.norm(module.weight, p=1)

        self.hook = self.register_forward_hook(conv_hook)
        self.regularization = 0

    def after_training(self):
        self.hook.remove()


if __name__ == "__main__":
    torch.manual_seed(229)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Prepare the demo network
    demo_net = DemoNet()
    demo_net.before_training()
    mnist.train_model(demo_net, epochs=100, batch_size=3000, regular=True)
    demo_net.after_training()
    mnist.save_param(demo_net, "demo_param.pkl")

    # One shot pruning with fine-tuning
    demo_net = DemoNet()
    mnist.load_param(demo_net, "demo_param.pkl")
    mnist.valid_model(demo_net, batch_size=2500)  # 94.97 / FLOPs 600064
    demo_data = Variable(torch.rand(1, 1, 32, 32), requires_grad=True)
    pm.one_shot_prune(demo_net.cpu(), demo_data, method="minimum_weight", prune_ratio=0.1)
    mnist.valid_model(demo_net, batch_size=2500)  # 92.02 / FLOPs 538624
    mnist.train_model(demo_net, epochs=10, batch_size=3000, regular=False)
    mnist.valid_model(demo_net, batch_size=2500)  # 95.89 / FLOPs 538624

    # Iterative pruning cut with fine-tuning
    demo_net = DemoNet()
    mnist.load_param(demo_net, "demo_param.pkl")
    demo_net.cuda()
    mnist.valid_model(demo_net, batch_size=2500)  # 94.97 / FLOPs 600064
    demo_data = Variable(torch.rand(1, 1, 32, 32), requires_grad=True)
    pm.iterative_prune(demo_net, demo_data, mnist.get_train_loader(2500), method="minimum_weight", prune_ratio=0.1)
    mnist.valid_model(demo_net, batch_size=2500)  # 94.90 / FLOPs 538624
