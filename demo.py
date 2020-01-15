import torch.nn as nn

import modules.abstract_network as an
import modules.basic_module as bm


class DemoNet(an.AbstractNetwork):
    def __init__(self):
        an.AbstractNetwork.__init__(self)

        self.layer_trunk = nn.ModuleList([
            bm.Conv2d(5, 4, 3, padding=1), bm.BatchNorm2d(4),
            bm.Conv2d(4, 4, 3, padding=1), bm.BatchNorm2d(4),
            bm.Conv2d(4, 5, 3, padding=1), bm.BatchNorm2d(5),
            bm.Conv2d(5, 4, 3, padding=1), bm.BatchNorm2d(4),
            bm.Conv2d(4, 4, 3, padding=1), bm.BatchNorm2d(4)
        ])
        self.activate = nn.ReLU()

    def forward(self, input_tensor):
        x1 = self.activate(self.layer_trunk[1](self.layer_trunk[0](input_tensor)))
        x2 = self.activate(self.layer_trunk[3](self.layer_trunk[2](x1)))
        x3 = self.activate(self.layer_trunk[5](self.layer_trunk[4](x2 + x1)))
        x4 = self.activate(self.layer_trunk[7](self.layer_trunk[6](x3 + input_tensor)))
        x5 = self.activate(self.layer_trunk[9](self.layer_trunk[8](x4 + x2 + x1)))
        return x5


if __name__ == "__main__":
    import utils.utils_torch as ut

    result = ut.network_analysis(DemoNet(), input_channel=5)
    # [[-1], [0], [1], [2], [3, 1], [4], [5, -1], [6], [7, 3, 1], [8]]
