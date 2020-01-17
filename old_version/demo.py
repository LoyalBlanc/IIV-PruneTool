import torch
import torch.nn as nn

import old_version.modules.abstract_network as an
import old_version.modules.basic_module as bm


class DemoLayer(an.AbstractNetwork):
    def __init__(self, ipc, opc):
        an.AbstractNetwork.__init__(self)
        self.c = bm.Conv2d(ipc, opc, 3, padding=1)
        self.b = bm.BatchNorm2d(opc)
        self.r = nn.ReLU()

    def forward(self, input_tensor):
        return self.r(self.b(self.c(input_tensor)))


class DemoNetwork(an.AbstractNetwork):
    def __init__(self):
        an.AbstractNetwork.__init__(self)
        self.layer1 = DemoLayer(1, 32)
        self.layer2 = DemoLayer(32, 64)
        self.layer3 = DemoLayer(64, 64)
        self.layer4 = DemoLayer(64, 32)
        self.layer5 = bm.Conv2d(32, 10, 3, padding=1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.network_analysis(1)

    def forward(self, input_tensor):
        x1 = self.layer1(input_tensor)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3 + x2)
        x5 = self.layer5(x4 + x1)
        return self.max_pooling(x5).squeeze_()


if __name__ == "__main__":
    import os
    from torch.optim import Adam
    import old_version.methods.abstract_method as pruning_methods
    import old_version.utils.utils_mnist as utils_mnist
    import old_version.utils.utils_torch as utils_torch

    torch.manual_seed(229)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    def basic_training(network, dataset, epochs, save_param=False):
        network.cuda()
        optimizer = Adam(demo_net.parameters())
        for epoch in range(epochs):
            step_loss = pruning_tool.train_model_once(demo_net, dataset, nn.CrossEntropyLoss(), optimizer)
            print('Epoch [{}/{}], Step Loss: {:.4f}'.format(epoch + 1, epochs, step_loss))
            if save_param and step_loss < 0.1:
                utils_torch.save_param(demo_net, "demo_param.pkl")


    """
    demo_flag:
    0: preparing for test
    1: one-shot pruning with 20 epochs fine-tuning
    2: 20 epochs iterative pruning
    3: 20 epochs automatic pruning
    """
    demo_flag = -1

    training_dataset = utils_mnist.get_train_loader(1000)
    demo_net = DemoNetwork()
    pruning_tool = pruning_methods.PruningTool(input_channel=1, pruning_rate=0.1)

    if demo_flag == 0:
        basic_training(demo_net, training_dataset, epochs=100, save_param=True)
        utils_mnist.valid_model(demo_net, batch_size=1000)  # 100 Epochs / Acc:94.65 / FLOPs:600064
    elif demo_flag == 1:
        utils_torch.load_param(demo_net, "demo_param_1.pkl")
        pruning_tool.one_shot_pruning(demo_net)
        utils_mnist.valid_model(demo_net, batch_size=1000)  # 100 Epochs / Acc:38.93 / FLOPs:538624
        basic_training(demo_net, training_dataset, 20)
        utils_mnist.valid_model(demo_net, batch_size=1000)  # 120 Epochs / Acc:93.34 / FLOPs:538624
    # elif demo_flag == 2:
    #     utils_torch.load_param(demo_net, "demo_param_1.pkl")
    #     pruning_tool.iterative_pruning(demo_net, training_dataset, nn.CrossEntropyLoss(), epoch_interval=2)
    #     utils_mnist.valid_model(demo_net, batch_size=1000)  # 120 Epochs / Acc:0 / FLOPs:0
    # elif demo_flag == 3:
    #     utils_torch.load_param(demo_net, "demo_param_1.pkl")
    #     pruning_tool.automatic_pruning(demo_net, training_dataset, nn.CrossEntropyLoss(), epochs=20)
    #     utils_mnist.valid_model(demo_net, batch_size=1000)  # 000 Epochs / Acc:0 / FLOPs:0
