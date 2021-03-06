import os

import torch
import torch.nn as nn
import torchvision.models as models

import pruning_tools as pt
import utils


def basic_training(network, dataset, epochs, lr=1e-3):
    network.cuda()
    for epoch in range(epochs):
        args = (dataset, nn.CrossEntropyLoss(), lr)
        step_loss, lr = pt.train_model_once(network, "minimum_weight", utils.train_one_epoch, *args)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, step_loss))


def basic_validating(network):
    return utils.valid_model(network, batch_size=1000, verbose=False)


if __name__ == "__main__":
    torch.manual_seed(229)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # -------------------------------------------------- #
    # MNIST Test
    # Pre-train 10 epochs & Automatic pruning 10 epochs
    # FLOPS      593920 ->  343524 (Remain 57.84%)
    # Accuracy   0.9921 ->  0.9918
    # -------------------------------------------------- #
    model = models.resnet18(pretrained=True)

    # Analysis the network
    dummy_data = torch.ones(1, 3, 96, 96)
    pt.analyze_network(model, dummy_data, verbose=False, for_pruning=True)

    # Pre-train the model
    training_dataset = utils.get_train_loader(2000)
    basic_training(model, training_dataset, 10)

    # Automatic pruning
    training_args = (training_dataset, nn.CrossEntropyLoss(), 1e-3)
    pt.automatic_pruning(model, dummy_data, basic_validating, 99, utils.train_one_epoch, *training_args, epochs=10)
