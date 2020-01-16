import os

import torch
import torch.nn as nn

import modules.abstract_network as an
import modules.basic_module as bm


class DemoNet(an.AbstractNetwork):
    def __init__(self):
        an.AbstractNetwork.__init__(self)

        self.conv1 = bm.Conv2d(1, 16, 3, padding=1)
        self.conv2 = bm.Conv2d(16, 32, 3, padding=1)
        self.conv3 = bm.Conv2d(32, 32, 3, padding=1)
        self.conv4 = bm.Conv2d(32, 16, 3, padding=1)
        self.conv5 = bm.Conv2d(16, 10, 3, padding=1)

        self.bn1 = bm.BatchNorm2d(16)
        self.bn2 = bm.BatchNorm2d(32)
        self.bn3 = bm.BatchNorm2d(32)
        self.bn4 = bm.BatchNorm2d(16)
        self.bn5 = bm.BatchNorm2d(10)

        self.activate = nn.ReLU()
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

        self.network_analysis(1)

    def forward(self, input_tensor):
        x1 = self.activate(self.bn1(self.conv1(input_tensor)))
        x2 = self.activate(self.bn2(self.conv2(x1)))
        x3 = self.activate(self.bn3(self.conv3(x2)))
        x4 = self.activate(self.bn4(self.conv4(x3 + x2)))
        x5 = self.activate(self.bn5(self.conv5(x4 + x1)))
        return self.max_pooling(x5).squeeze_()


if __name__ == "__main__":
    import method.abstract_method as am
    import utils.utils_mnist as um

    torch.manual_seed(229)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    """
    demo_flag:
    0: preparing for test
    1: one-shot pruning
    2: iterative pruning
    """
    demo_flag = 0
    demo_net = DemoNet()

    if demo_flag == 0:
        am.before_training(demo_net, "minimum_weight")
        um.train_model(demo_net, epochs=10, batch_size=5000, regular=False)
        am.after_training(demo_net, "minimum_weight")
        um.valid_model(demo_net, batch_size=5000)
        um.save_param(demo_net, "demo_param.pkl")
    elif demo_flag == 1:
        um.load_param(demo_net, "demo_param.pkl")
        am.one_shot_pruning(demo_net, 1, "minimum_weight", 0.1)
        um.valid_model(demo_net, batch_size=5000)
