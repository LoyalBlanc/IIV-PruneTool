import copy

import torch
import torch.nn as nn
from torch.optim import Adam

import PruningMethod.minimum_weight as mw
from utils.utils import get_model_flops


def get_the_method_library(method):
    if method == "minimum_weight":
        return mw
    else:
        # Todo: more pruning method will be added in the future.
        pass


def prepare_pruning(network, example_data, method, prune_ratio):
    network.cuda()
    example_data = torch.zeros(1, 1, 32, 32).cuda() \
        if example_data is None else example_data.cuda()
    method_library = get_the_method_library(method)
    flops_now = get_model_flops(network, example_data)
    flops_target = int(flops_now * (1 - prune_ratio))
    print("FLOPs before pruning is %d, target is %d." % (flops_now, flops_target))
    return example_data, method_library, flops_now, flops_target


def training_with_regularization(network, dataset, criterion, optimizer, method_library):
    sum_loss = 0
    method_library.before_training(network)
    for index, (images, labels) in enumerate(dataset):
        outputs = network(images.cuda())
        loss = criterion(outputs, labels.cuda()) + 1e-3 * network.regularization
        network.regularization = 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    method_library.after_training(network)
    return sum_loss


def one_shot_prune(network,
                   example_data=None,
                   method="minimum_weight",
                   prune_ratio=0.1):
    print("Start one-shot pruning.")
    example_data, method_library, flops_now, flops_target = prepare_pruning(network, example_data, method, prune_ratio)

    while flops_now > flops_target:
        method_library.prune_one_channel(network)
        flops_now = get_model_flops(network, example_data)
    print("Successfully prune the network, the FLOPs now is %d" % flops_now)
    return network


def iterative_prune(network,
                    dataset,
                    example_data=None,
                    method="minimum_weight",
                    prune_ratio=0.1,
                    criterion=nn.MSELoss(),
                    lr=1e-3):
    print("Start iterative pruning.")
    example_data, method_library, flops_now, flops_target = prepare_pruning(network, example_data, method, prune_ratio)

    network.train()
    optimizer = Adam(network.parameters(), lr=lr)
    while flops_now > flops_target:
        method_library.prune_one_channel(network)
        flops_now = get_model_flops(network, example_data)
        sum_loss = training_with_regularization(network, dataset, criterion, optimizer, method_library)
        print('FLOPs: %d,  Loss: %.4f' % (flops_now, sum_loss))
    print("Successfully prune the network, the FLOPs now is %d" % flops_now)
    return network


def automotive_prune(network,
                     dataset,
                     example_data=None,
                     method="minimum_weight",
                     prune_ratio=0.1,
                     criterion=nn.MSELoss(),
                     lr=1e-3,
                     epoch_limit=10,
                     step_loss_decay=0.5):
    print("Start automotive pruning.")
    example_data, method_library, flops_now, flops_target = prepare_pruning(network, example_data, method, prune_ratio)

    network_backup = copy.deepcopy(network)
    network.train()
    optimizer = Adam(network.parameters(), lr=lr)
    step_loss = training_with_regularization(network, dataset, criterion, optimizer, method_library)
    epoch_count = 0
    while flops_now > flops_target:
        sum_loss = training_with_regularization(network, dataset, criterion, optimizer, method_library)
        if sum_loss < step_loss:
            epoch_count = 0
            step_loss = sum_loss * (1 - step_loss_decay) + step_loss * step_loss_decay
            network_backup = copy.deepcopy(network)
            method_library.prune_one_channel(network)
            flops_now = get_model_flops(network, example_data)
            print('Loss: %.4f, Step Loss: %.4f, new FLOPs: %d' % (sum_loss, step_loss, flops_now))
        elif epoch_count < epoch_limit:
            epoch_count += 1
            print('Loss: %.4f, Step Loss: %.4f' % (sum_loss, step_loss))
        else:
            flops_now = get_model_flops(network_backup, example_data)
            print("Exceed the epoch limit and terminate pruning, the FLOPs now is %d" % flops_now)
            break
    print("Successfully prune the network, the FLOPs now is %d" % flops_now)
    return network_backup


if __name__ == "__main__":
    pass
