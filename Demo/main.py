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
    train_model(training_network, epochs=5)

    pruning_network = DemoNetworkForPruning(Basic)
    pruning_network.load_state_dict(training_network.state_dict())
    valid_model(pruning_network)

    OneShot(0.01).prune(pruning_network)
    print(pruning_network.get_pruned_channel())
    valid_model(pruning_network)
    train_model(pruning_network, epochs=1)
    valid_model(pruning_network)
