from . import utils
from . import minimum_weight as mw
import torch
import torch.optim as op


def get_the_module_name(method):
    if method == "minimum_weight":
        return mw
    else:
        # Todo: More pruning modules will be added in the future.
        pass


def train_model_once(model, dataset, criterion, lr, method):
    method_module = get_the_module_name(method)
    method_module.before_training(model)
    model.cuda()
    model.train()
    optimizer = op.Adam(model.parameters(), lr=lr)
    step_loss = 0
    for index, (images, labels) in enumerate(dataset):
        outputs = model(torch.cat((images, images, images), dim=1).cuda())
        loss = criterion(outputs, labels.cuda()) + 1e-3 * model.regularization
        model.regularization = 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss += loss.item()
    method_module.after_training(model)
    lr = optimizer.param_groups[0]['lr']
    model.eval()
    model.cpu()
    return step_loss, lr


def _prepare_pruning(network, example_data, method, pruning_rate):
    network.cuda()
    network.eval()
    flops_now = utils.get_model_flops(network, example_data)
    flops_target = int(flops_now * (1 - pruning_rate))
    print("FLOPs before pruning is %d, target is %d." % (flops_now, flops_target))
    method_module = get_the_module_name(method)
    method_module.prepare_pruning(network)
    network.cpu()
    return flops_target, method_module


def one_shut_pruning(network, example_data, method, pruning_rate):
    example_data = example_data.cuda()
    print("Start one-shot pruning.")
    flops_target, method_module = _prepare_pruning(network, example_data, method, pruning_rate)
    flops_now = flops_target + 1

    while flops_now > flops_target:
        network.cpu()
        method_module.prune_network_once(network)
        network.cuda()
        flops_now = utils.get_model_flops(network, example_data)
        # from utils.auxiliary_dataset import valid_model
        # valid_model(network, batch_size=2500)
    print("Successfully prune the network, the FLOPs now is %d" % flops_now)
    return network


def iterative_pruning(network, dataset, example_data, method, pruning_rate, criterion, lr):
    example_data = example_data.cuda()
    print("Start iterative pruning.")
    flops_target, method_module = _prepare_pruning(network, example_data, method, pruning_rate)
    flops_now = flops_target + 1

    epoch = 0
    while flops_now > flops_target:
        network.cuda()
        flops_now = utils.get_model_flops(network, example_data)
        step_loss, lr = train_model_once(network, dataset, criterion, lr, "minimum_weight")
        epoch += 1
        print('Epoch {},  Loss: {:.4f}'.format(epoch, step_loss))
        # from utils.auxiliary_dataset import valid_model
        # valid_model(network, batch_size=2500)
        network.cpu()
        method_module.prune_network_once(network)

    print("Successfully prune the network, the FLOPs now is %d" % flops_now)
    return network
