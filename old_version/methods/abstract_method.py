import methods.minimum_weight as mw
import torch
import utils.utils_torch as utils_torch


class PruningTool(object):
    def __init__(self, input_channel, method="minimum_weight", pruning_rate=0.1):
        self._example_data = torch.rand(1, input_channel, 32, 32).cuda()
        self._method = self._get_the_library(method)
        self.pruning_rate = pruning_rate

    @staticmethod
    def _get_the_library(method):
        if method == "minimum_weight":
            return mw
        else:
            # Todo: more pruning methods will be added in the future.
            pass

    def update_input_channel(self, input_channel):
        self._example_data = torch.rand(1, input_channel, 32, 32).cuda()

    def _prepare_pruning(self, network):
        network.cuda()
        network.eval()
        flops_now = utils_torch.get_model_flops(network, self._example_data)
        flops_target = int(flops_now * (1 - self.pruning_rate))
        print("FLOPs before pruning is %d, target is %d." % (flops_now, flops_target))
        self._method.prepare_pruning(network)
        network.cpu()
        return flops_target

    def one_shot_pruning(self, network):
        print("Start one-shot pruning.")
        flops_target = self._prepare_pruning(network)
        flops_now = flops_target + 1
        while flops_now > flops_target:
            network.cpu()
            self._method.prune_network_once(network)
            network.cuda()
            flops_now = utils_torch.get_model_flops(network, self._example_data)
        print("Successfully prune the network, the FLOPs now is %d" % flops_now)
        return network

    # def iterative_pruning(self, network, dataset, criterion, lr=1e-3, epoch_interval=1):
    #     network.cuda()
    #     optimizer = Adam(network.parameters(), lr=lr)
    #     print("Start iterative pruning.")
    #     flops_target = self._prepare_pruning(network)
    #     flops_now = flops_target + 1
    #     epoch = 0
    #     while flops_now > flops_target:
    #         self._method.prune_network_once(network)
    #         flops_now = utils_torch.get_model_flops(network, self._example_data)
    #         for _ in range(epoch_interval):
    #             epoch += 1
    #             step_loss = self.train_model_once(network, dataset, criterion, optimizer)
    #             print("Epoch %d, FLOPs: %d, Step Loss: %.4f" % (epoch, flops_now, step_loss))
    #
    #     print("Successfully prune the network, the FLOPs now is %d" % flops_now)
    #     return network
    #
    # def automatic_pruning(self, network, dataset, criterion, lr=1e-3, epochs=10):
    #     network.cuda()
    #     optimizer = Adam(network.parameters(), lr=lr)
    #     print("Start automatic pruning.")
    #     self._method.prepare_pruning(network)
    #     flops_now = utils_torch.get_model_flops(network, self._example_data)
    #     loss_limit = 0
    #     network_backup = copy.deepcopy(network)
    #     for epoch in range(epochs):
    #         step_loss = self.train_model_once(network, dataset, criterion, optimizer)
    #         if loss_limit == 0:
    #             loss_limit = step_loss
    #         elif loss_limit > step_loss:
    #             loss_limit = step_loss
    #             network_backup = copy.deepcopy(network)
    #             self._method.prune_network_once(network)
    #             network.cuda()
    #             flops_now = utils_torch.get_model_flops(network, self._example_data)
    #         print("Epoch %d, FLOPs: %d, Step Loss: %.4f, Loss Limit: %.4f"
    #               % (epoch, flops_now, step_loss, loss_limit))
    #     network_backup.cuda()
    #     flops_now = utils_torch.get_model_flops(network_backup, self._example_data)
    #     print("Successfully prune the network, the FLOPs now is %d" % flops_now)
    #     return network_backup

    def train_model_once(self, network, dataset, criterion, optimizer):
        self._method.before_training(network)
        network.cuda()
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
        self._method.after_training(network)
        network.eval()
        network.cpu()
        return step_loss
