import os
from Demo.Model import BasicConv, DemoNetworkForTraining, DemoNetworkForPruning
from Demo.Function import train_model, valid_model
from Module.ModuleParameter.MinimumWeight import MinimumWeight

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def prune(model, ratio):
    model.calculate_network_contribution()
    network_score, network_score_index = model.get_network_contribution()
    prune_limit = int(ratio * network_score.shape[0])
    prune_index = network_score_index[network_score.sort(dim=0)[1][:prune_limit]]
    _, prune_index_sorted = prune_index[:, 1].sort(dim=0, descending=True)
    for index in prune_index_sorted:
        conv_index, channel_index = prune_index[index]
        model.prune_index(conv_index, channel_index)
    print(model.get_pruned_channel())


if __name__ == "__main__":
    # Origin model
    training_network = DemoNetworkForTraining(BasicConv)
    valid_model(training_network)  # 10.28
    train_model(training_network, epochs=10)
    valid_model(training_network)  # 98.31

    # One shot pruning
    pruning_network = DemoNetworkForPruning(MinimumWeight)
    pruning_network.load_state_dict(training_network.state_dict())
    pruning_network.before_pruning_network()
    train_model(pruning_network, epochs=5, regular=True)
    prune(pruning_network, 0.41)
    pruning_network.after_pruning_network()
    train_model(pruning_network, epochs=15, regular=False)
    valid_model(pruning_network)  # 99.06 / 34, 30, 41, 37, 47

    # Iterative pruning
    pruning_network = DemoNetworkForPruning(MinimumWeight)
    pruning_network.load_state_dict(training_network.state_dict())
    pruning_network.before_pruning_network()
    train_model(pruning_network, epochs=5, regular=True)
    for _ in range(5):
        prune(pruning_network, 0.1)
        train_model(pruning_network, epochs=2, regular=True)
    pruning_network.after_pruning_network()
    train_model(pruning_network, epochs=5, regular=False)
    valid_model(pruning_network)  # 98.40 / 29, 31, 47, 46, 37
