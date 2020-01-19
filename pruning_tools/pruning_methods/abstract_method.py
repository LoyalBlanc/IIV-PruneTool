import copy

from . import minimum_weight as mw
from . import utils


def get_the_module_name(method):
    # Todo: More pruning modules will be added in the future.
    if method == "minimum_weight":
        return mw
    else:
        return


def train_model_once(model, method, func_train_one_epoch, *training_args):
    method_module = get_the_module_name(method)
    method_module.before_training(model)
    model.train()
    step_loss, lr = func_train_one_epoch(model, *training_args)
    method_module.after_training(model)
    return step_loss, lr


def _prepare_pruning(network, example_data, method, pruning_rate):
    network.cuda()
    flops_now = utils.get_model_flops(network, example_data)
    flops_target = int(flops_now * (1 - pruning_rate))
    print("FLOPs before pruning is %d, target is %d." % (flops_now, flops_target))
    method_module = get_the_module_name(method)
    method_module.prepare_pruning(network, example_data)
    network.cpu()
    return flops_now, flops_target, method_module


def one_shut_pruning(network,
                     example_data,
                     method="minimum_weight",
                     pruning_rate=0.1):
    example_data = example_data.cuda()
    print("Start one-shot pruning.")
    flops_now, flops_target, method_module = _prepare_pruning(network, example_data, method, pruning_rate)

    while flops_now > flops_target:
        network.cpu()
        method_module.prune_network_once(network)
        network.cuda()
        flops_now = utils.get_model_flops(network, example_data)

    print("Successfully prune the network, the FLOPs now is %d" % flops_now)
    return network


def iterative_pruning(network,
                      example_data,
                      func_train_one_epoch,
                      *training_args,
                      method="minimum_weight",
                      pruning_rate=0.1,
                      pruning_interval=1):
    example_data = example_data.cuda()
    print("Start iterative pruning.")
    flops_now, flops_target, method_module = _prepare_pruning(network, example_data, method, pruning_rate)

    epoch = 0
    while flops_now > flops_target:
        network.cuda()
        flops_now = utils.get_model_flops(network, example_data)
        if pruning_interval < 1:
            interval = int(1 / pruning_interval)
            epoch += 1
            step_loss, lr = train_model_once(network, method, func_train_one_epoch, *training_args)
            training_args = (training_args[0], training_args[1], lr)
            print('Epoch {}, Loss: {:.4f}, FLOPs: {}'.format(epoch, step_loss, flops_now))
            network.cpu()
            for _ in range(interval):
                method_module.prune_network_once(network)
        else:
            interval = int(pruning_interval)
            for _ in range(interval):
                epoch += 1
                step_loss, lr = train_model_once(network, method, func_train_one_epoch, *training_args)
                training_args = (training_args[0], training_args[1], lr)
                print('Epoch {}, Loss: {:.4f}, FLOPs: {}'.format(epoch, step_loss, flops_now))
            network.cpu()
            method_module.prune_network_once(network)

    print("Successfully prune the network, the FLOPs now is %d" % flops_now)
    return network


def automatic_pruning(network,
                      example_data,
                      func_valid,
                      target_accuracy,
                      func_train_one_epoch,
                      *training_args,
                      method="minimum_weight",
                      epochs=10, ):
    example_data = example_data.cuda()
    print("Start automatic pruning.")
    network.cuda()
    method_module = get_the_module_name(method)
    method_module.prepare_pruning(network, example_data)

    network_backup = copy.deepcopy(network)
    for epoch in range(epochs):
        network.cuda()
        step_loss, lr = train_model_once(network, method, func_train_one_epoch, *training_args)
        training_args = (training_args[0], training_args[1], lr)
        acc = func_valid(network)
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}'.format(epoch + 1, epochs, step_loss, acc))
        if acc > target_accuracy:
            while acc > target_accuracy:
                network_backup = copy.deepcopy(network)
                network.cpu()
                method_module.prune_network_once(network)
                acc = func_valid(network)
            flops_now = utils.get_model_flops(network, example_data)
            print("Update network backup, FLOPs: %d" % flops_now)

    network_backup.cuda()
    flops_now = utils.get_model_flops(network_backup, example_data)
    print("Successfully prune the network, the FLOPs now is %d" % flops_now)
    return network_backup
