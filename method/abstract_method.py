import torch
from torch.optim import Adam

import method.minimum_weight as mw
import utils.utils_torch as utils_torch


class PruningTool(object):
    def __init__(self, input_channel, method="minimum_weight", pruning_rate=0.1):
        self.example_data = torch.rand(1, input_channel, 32, 32).cuda()
        self.method = self.get_the_library(method)
        self.pruning_rate = pruning_rate

    @staticmethod
    def get_the_library(method):
        if method == "minimum_weight":
            return mw
        else:
            # Todo: more pruning method will be added in the future.
            pass

    def update_input_channel(self, input_channel):
        self.example_data = torch.rand(1, input_channel, 32, 32).cuda()

    def prepare_pruning(self, network):
        network.cuda()
        network.eval()
        flops_now = utils_torch.get_model_flops(network, self.example_data)
        flops_target = int(flops_now * (1 - self.pruning_rate))
        print("FLOPs before pruning is %d, target is %d." % (flops_now, flops_target))
        self.method.before_pruning(network)
        return flops_target

    def one_shot_pruning(self, network):
        print("Start one-shot pruning.")
        flops_target = self.prepare_pruning(network)
        flops_now = flops_target + 1
        while flops_now > flops_target:
            network.cpu()
            self.method.prune_once(network)
            network.cuda()
            flops_now = utils_torch.get_model_flops(network, self.example_data)
        print("Successfully prune the network, the FLOPs now is %d" % flops_now)
        return network

    def iterative_pruning(self, network, dataset, criterion, lr=1e-3, epoch_interval=1):
        optimizer = Adam(network.parameters(), lr=lr)
        print("Start iterative pruning.")
        flops_target = self.prepare_pruning(network)
        flops_now = flops_target + 1
        while flops_now > flops_target:
            network.cpu()
            self.method.prune_once(network)
            network.cuda()
            flops_now = utils_torch.get_model_flops(network, self.example_data)
            for _ in range(epoch_interval):
                step_loss = self.training_once(network, dataset, criterion, optimizer)
                print("FLOPs: %d, Step Loss: %.4f" % (flops_now, step_loss))
        print("Successfully prune the network, the FLOPs now is %d" % flops_now)
        return network

    def automatic_pruning(self, network, dataset, criterion, lr=1e-3, epochs=10):
        optimizer = Adam(network.parameters(), lr=lr)
        print("Start automatic pruning.")
        self.prepare_pruning(network)
        flops_now = 0
        for epoch in range(epochs):
            # Todo: Pruning the network automatically
            network.cpu()
            self.method.prune_once(network)
            network.cuda()
            flops_now = utils_torch.get_model_flops(network, self.example_data)
            self.training_once(network, dataset, criterion, optimizer)
        print("Successfully prune the network, the FLOPs now is %d" % flops_now)
        return network

    def training_once(self, network, dataset, criterion, optimizer):
        self.method.before_training(network)
        network.train()
        step_loss = 0
        for index, (images, labels) in enumerate(dataset):
            outputs = network(images.cuda())
            loss = criterion(outputs, labels.cuda()) + 1e-3 * network.regularization
            network.regularization = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_loss += loss.item()
        self.method.after_training(network)
        network.eval()
        return step_loss
