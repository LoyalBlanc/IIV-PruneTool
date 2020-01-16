import torch

import method.minimum_weight as mw
import utils.utils_torch as ut


def get_the_library(method):
    if method == "minimum_weight":
        return mw
    else:
        # Todo: more pruning method will be added in the future.
        pass


def before_training(network, method):
    method_library = get_the_library(method)
    method_library.before_training(network)


def after_training(network, method):
    method_library = get_the_library(method)
    method_library.after_training(network)


def prepare_pruning(network, input_channel, method, prune_ratio):
    network.cuda()
    example_data = torch.rand(1, input_channel, 32, 32).cuda()
    flops_now = ut.get_model_flops(network, example_data)
    flops_target = int(flops_now * (1 - prune_ratio))
    print("FLOPs before pruning is %d, target is %d." % (flops_now, flops_target))

    method_library = get_the_library(method)
    method_library.before_pruning(network)
    return example_data, method_library, flops_now, flops_target


def one_shot_pruning(network, input_channel, method, pruning_rate):
    print("Start one-shot pruning.")
    example_data, method_library, flops_now, flops_target = \
        prepare_pruning(network, input_channel, method, pruning_rate)

    while flops_now > flops_target:
        method_library.prune_one_channel(network)
        flops_now = ut.get_model_flops(network, example_data)
    print("Successfully prune the network, the FLOPs now is %d" % flops_now)
    return network
