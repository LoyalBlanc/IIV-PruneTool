import os

import torch
import torch.nn as nn
import torch.optim as op
import torchvision.models as models

import pruning_tools as pt
from data import utils


def train_one_epoch(model, dataset, criterion, lr):
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
    return step_loss, optimizer.param_groups[0]['lr']


def basic_training(network, dataset, epochs, lr=1e-3):
    network.cuda()
    for epoch in range(epochs):
        args = (dataset, nn.CrossEntropyLoss(), lr)
        step_loss, lr = pt.train_model_once(network, "minimum_weight", train_one_epoch, *args)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, step_loss))


if __name__ == "__main__":
    torch.manual_seed(229)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    """
    demo_flag:
    0: preparing for test
    1: one-shot pruning with fine-tuning
    2: iterative pruning
    3: automatic pruning (recommended)
    """
    demo_flag = 0
    demo_model = models.resnet18()
    dummy_data = torch.ones(1, 3, 64, 64)
    pt.analyze_network(demo_model, dummy_data, verbose=False, for_pruning=True)
    training_dataset = utils.get_train_loader(4000)

    if demo_flag == 0:
        basic_training(demo_model, training_dataset, 10)
        utils.valid_model(demo_model, batch_size=5000)  # Acc 99.21 / FLOPs 593920
        utils.save_param(demo_model, "data/demo_param.pkl")
    elif demo_flag == 1:
        utils.load_param(demo_model, "data/demo_param_.pkl")
        pt.one_shut_pruning(demo_model, dummy_data, method="minimum_weight", pruning_rate=0.2)
        utils.valid_model(demo_model, batch_size=5000)  # Acc 98.49 / FLOPs 474112
    elif demo_flag == 2:
        utils.load_param(demo_model, "data/demo_param_.pkl")
        training_args = (training_dataset, nn.CrossEntropyLoss(), 1e-3)
        pt.iterative_pruning(demo_model, dummy_data, train_one_epoch, *training_args,
                             pruning_rate=0.2, pruning_interval=0.1)
        utils.valid_model(demo_model, batch_size=5000)  # Acc 98.81 / FLOPs 455680
    elif demo_flag == 3:
        utils.load_param(demo_model, "data/demo_param_.pkl")
        training_args = (training_dataset, nn.CrossEntropyLoss(), 1e-3)
        pt.automatic_pruning(demo_model, dummy_data, utils.valid_model, 99, train_one_epoch, *training_args, epochs=200)
        utils.valid_model(demo_model, batch_size=5000)  # Acc 99.19 / FLOPs 343524
