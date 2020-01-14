import torch.nn as nn
from torch.optim import Adam

import PruningMethod.minimum_weight as mw
from utils import get_model_flops


def prune_network(network, method):
    # network
    if method == "minimum_weight":
        mw.prune_one_channel(network)
    else:
        # Todo: more pruning method will be added in the future.
        pass


def one_shot_prune(network, example_data, method, prune_ratio):
    example_data = example_data.cuda()
    network.cuda()
    flops_now = get_model_flops(network, example_data)
    flops_target = int(flops_now * (1 - prune_ratio))
    print("FLOPs before pruning is %d, target is %d." % (flops_now, flops_target))
    try:
        while flops_now > flops_target:
            prune_network(network, method)
            flops_now = get_model_flops(network, example_data)
    except RuntimeError:
        print("Fail to prune the network, one layer lost all the channels.")
    except Exception as e:
        print("Fatal Error!\n %s" % e)
    else:
        print("Successfully prune the network, the FLOPs now is %d" % flops_now)


def iterative_prune(network, example_data, dataset, method, prune_ratio, lr=1e-3):
    example_data = example_data.cuda()
    network.cuda()
    flops_now = get_model_flops(network, example_data)
    flops_target = int(flops_now * (1 - prune_ratio))
    print("FLOPs before pruning is %d, target is %d." % (flops_now, flops_target))

    network.train()
    optimizer = Adam(network.parameters(), lr=lr)
    while flops_now > flops_target:
        prune_network(network, method)
        flops_now = get_model_flops(network, example_data)

        network.before_training()
        sum_loss = 0
        for index, (images, labels) in enumerate(dataset):
            outputs = network(images.cuda())
            loss = nn.CrossEntropyLoss()(outputs, labels.cuda()) + 1e-3 * network.regularization
            network.regularization = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        print('FLOPs: %d,  Loss: %.4f' % (flops_now, sum_loss))
        network.after_training()


if __name__ == "__main__":
    pass
