import os
import torch
import torch.nn as nn
import torchvision.models as models
import network_analyzer as nwa
import pruning_methods as pmt
import utils


def basic_training(network, dataset, epochs, lr=1e-3):
    network.cuda()
    for epoch in range(epochs):
        step_loss, lr = pmt.train_model_once(network, dataset, nn.CrossEntropyLoss(), lr, "minimum_weight")
        print('Epoch [{}/{}],  Loss: {:.4f}'.format(epoch + 1, epochs, step_loss))


if __name__ == "__main__":
    torch.manual_seed(229)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    """
    demo_flag:
    0: preparing for test
    1: one-shot pruning with fine-tuning
    2: iterative pruning
    3: automatic pruning
    """
    demo_flag = 2
    demo_model = models.resnet18()
    dummy_data = torch.ones(1, 3, 64, 64)
    nwa.analyze_network(demo_model, dummy_data, verbose=False, for_pruning=True)
    training_dataset = utils.get_train_loader(3000)

    if demo_flag == 0:
        basic_training(demo_model, training_dataset, 10)
        utils.valid_model(demo_model, batch_size=2500)  # Acc 99.21 / FLOPs 593920
        utils.save_param(demo_model, "demo_param.pkl")
    elif demo_flag == 1:
        utils.load_param(demo_model, "demo_param.pkl")
        pmt.one_shut_pruning(demo_model, dummy_data, method="minimum_weight", pruning_rate=0.2)
        utils.valid_model(demo_model, batch_size=2500)  # Acc 98.20 / FLOPs 5807872
    elif demo_flag == 2:
        utils.load_param(demo_model, "demo_param.pkl")
        pmt.iterative_pruning(demo_model, training_dataset, dummy_data, method="minimum_weight",
                              pruning_rate=0.2, criterion=nn.CrossEntropyLoss(), lr=1e-3)
        utils.valid_model(demo_model, batch_size=2500)  # Acc 98.20 / FLOPs 5807872
