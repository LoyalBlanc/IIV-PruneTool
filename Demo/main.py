import os
import torch
from Demo.Model import BasicConv, DemoNetworkForTraining, DemoNetworkForPruning
from Demo.Train import train_model, valid_model
from Module.MinimumWeight.MinimumWeight import Basic
from Frame.OneShot import OneShot

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.manual_seed(229)

if __name__ == "__main__":
    training_network = DemoNetworkForTraining(BasicConv)
    valid_model(training_network)  # 8.59

    train_model(training_network, epochs=5)
    valid_model(training_network)  # 89.77

    pruning_network = DemoNetworkForPruning(Basic)
    pruning_network.load_state_dict(training_network.state_dict())
    OneShot(0.41).prune(pruning_network)
    train_model(pruning_network, epochs=5)
    valid_model(pruning_network)  # 97.85 / 37, 37, 38, 36, 41
    print(pruning_network.get_pruned_channel())

    pruning_network = DemoNetworkForPruning(Basic)
    pruning_network.load_state_dict(training_network.state_dict())
    for _ in range(5):
        OneShot(0.1).prune(pruning_network)
        train_model(pruning_network, epochs=1)
    valid_model(pruning_network)  # 66.69 / 39, 37, 38, 40, 36
    print(pruning_network.get_pruned_channel())
