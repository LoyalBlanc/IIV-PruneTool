import torch
import torch.nn as nn


class GbnModule(nn.Module):
    def __init__(self, ipc, opc, stride=1):
        super().__init__()
        self.ipc = ipc
        self.opc = opc
        self.stride = stride
        self.conv = nn.Conv2d(self.ipc, self.opc, 3, stride=self.stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.opc)
        self.relu = nn.ReLU(inplace=True)

        self.fai = nn.Parameter(torch.ones(1, self.opc, 1, 1), requires_grad=False)
        self.score = 0
        self.hook = None

    def forward(self, x):
        x = self.conv(x)
        x = self.fai * self.bn(x)
        return self.relu(x)

    def calculate_score(self, grad):
        self.score += (grad * self.fai).abs()

    def freeze(self):
        with torch.no_grad():
            self.fai.set_(self.bn.weight.view(1, -1, 1, 1))
            self.fai.requires_grad = True
            self.bn.bias.set_(torch.clamp(self.bn.bias / self.bn.weight, -10, 10))
            self.bn.weight.set_(torch.ones_like(self.bn.weight))
            self.bn.weight.requires_grad = False
        self.hook = self.fai.register_hook(self.calculate_score)

    def prune_ipc(self, prune_ipc_index):
        self.ipc -= 1
        conv_weight = torch.cat((self.conv.weight[:, 0:prune_ipc_index],
                                 self.conv.weight[:, prune_ipc_index + 1:]), dim=1)
        self.conv = nn.Conv2d(self.ipc, self.opc, 3, stride=self.stride, padding=1, bias=False)
        self.conv.weight = nn.Parameter(conv_weight)

    def prune_opc(self, prune_opc_index):
        self.opc -= 1
        conv_weight = torch.cat((self.conv.weight[0:prune_opc_index],
                                 self.conv.weight[prune_opc_index + 1:]), dim=0)
        self.conv = nn.Conv2d(self.ipc, self.opc, 3, stride=self.stride, padding=1, bias=False)
        self.conv.weight = nn.Parameter(conv_weight)

        bn_bias = torch.cat((self.bn.bias[0:prune_opc_index], self.bn.bias[prune_opc_index + 1:]), dim=0)
        self.bn = nn.BatchNorm2d(self.opc)
        self.bn.weight = nn.Parameter(torch.ones(self.opc), requires_grad=False)
        self.bn.bias = nn.Parameter(bn_bias)

        self.fai = nn.Parameter(torch.cat((self.fai[0:1, 0:prune_opc_index],
                                           self.fai[0:1, prune_opc_index + 1:]), dim=1))

        return prune_opc_index

    def melt(self):
        with torch.no_grad():
            self.bn.bias.set_(self.bn.bias * self.fai.view(-1))
            self.bn.weight.set_(self.fai.view(-1))
            self.bn.weight.requires_grad = True
            self.fai = nn.Parameter(torch.ones(1, self.opc, 1, 1), requires_grad=False)
        self.hook.remove()
        self.hook = None


if __name__ == "__main__":
    test_conv = GbnModule(3, 4, 2)
    print(test_conv(torch.rand([2, 3, 16, 32])).shape)

    test_conv.freeze()
    test_output = test_conv(torch.rand([2, 3, 16, 32]))
    print(test_output.shape)
    test_output.sum().backward()

    test_conv.prune_ipc(0)
    print(test_conv(torch.rand([2, 2, 16, 32])).shape)

    print(test_conv.get_min_score(), test_conv.prune_opc())
    print(test_conv(torch.rand([2, 2, 16, 32])).shape)

    test_conv.melt()
    print(test_conv(torch.rand([2, 2, 16, 32])).shape)
