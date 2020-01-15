import torch
import torch.nn as nn


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def prune_ipc(self, prune_ipc_index):
        self.in_channels -= 1
        conv_weight = torch.cat((self.weight[:, 0:prune_ipc_index], self.weight[:, prune_ipc_index + 1:]), dim=1)
        self.weight = nn.Parameter(conv_weight)

    def prune_opc(self, prune_opc_index):
        self.out_channels -= 1
        conv_weight = torch.cat((self.weight[0:prune_opc_index], self.weight[prune_opc_index + 1:]), dim=0)
        self.weight = nn.Parameter(conv_weight)
        if self.bias is not None:
            conv_bias = torch.cat((self.bias[0:prune_opc_index], self.bias[prune_opc_index + 1:]), dim=0)
            self.bias = nn.Parameter(conv_bias)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False):
        nn.BatchNorm2d.__init__(self, num_features, eps, momentum, affine, track_running_stats)

    def prune_features_num(self, prune_fea_index):
        self.num_features -= 1
        bn_weight = torch.cat((self.weight[0:prune_fea_index], self.weight[prune_fea_index + 1:]), dim=0)
        bn_bias = torch.cat((self.bias[0:prune_fea_index], self.bias[prune_fea_index + 1:]), dim=0)
        self.weight = nn.Parameter(bn_weight)
        self.bias = nn.Parameter(bn_bias)


class ReLU(nn.ReLU):
    def __init__(self, inplace=False):
        nn.ReLU.__init__(self, inplace)


class BasicModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        nn.Module.__init__(self)
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn = BatchNorm2d(out_channels)
        self.relu = ReLU()

    def forward(self, features):
        return self.relu(self.bn(self.conv(features)))

    def prune_ipc(self, prune_ipc_index):
        self.conv.prune_ipc(prune_ipc_index)

    def prune_opc(self, prune_opc_index):
        self.conv.prune_opc(prune_opc_index)
        self.bn.prune_features_num(prune_opc_index)


if __name__ == "__main__":
    from torch import rand, Size

    test_data = rand(1, 5, 16, 16)
    test_module = BasicModule(5, 4)
    assert test_module(test_data).shape == Size([1, 4, 16, 16])

    test_data = rand(1, 4, 16, 16)
    test_module.prune_ipc(1)
    assert test_module(test_data).shape == Size([1, 4, 16, 16])

    test_module.prune_opc(2)
    assert test_module(test_data).shape == Size([1, 3, 16, 16])

    print("Pass the unit exam!")
