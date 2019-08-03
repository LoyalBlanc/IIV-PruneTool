from Dataset import DataPreFetcher
from Network import PruningNetwork
import torch
import torch.nn as nn


def network_train(train_network):
    print(0, valid(train_network.cuda()))
    train_loader = DataPreFetcher(6000, train=True)
    optimizer = torch.optim.Adam(train_network.parameters(), lr=1e-2)
    loss = nn.CrossEntropyLoss()
    train_step(train_loader, train_network, 20, optimizer, loss)
    for _ in range(8):
        train_network.prune(4)
        train_step(train_loader, train_network, 5, optimizer, loss)
    train_step(train_loader, train_network, 10, optimizer, loss)


def train_step(train_loader, train_network, epochs, optimizer, loss):
    train_network.cuda()
    for epoch in range(epochs):
        train_loader.refresh()
        train_images, train_labels = train_loader.next()
        iteration = 0
        while train_images is not None:
            iteration += 1
            outputs = train_network(train_images)
            step_loss = loss(outputs, train_labels)
            optimizer.zero_grad()
            step_loss.backward()
            optimizer.step()
            train_images, train_labels = train_loader.next()
            print(-1, epoch + 1, iteration, step_loss.item())
        print(epoch + 1, valid(train_network))


def valid(valid_network):
    valid_loader = DataPreFetcher(10000, train=False)
    valid_network.eval()
    with torch.no_grad():
        valid_loader.refresh()
        valid_images, valid_labels = valid_loader.next()
        outputs = valid_network(valid_images)
        _, predicted = torch.max(outputs.data, 1)
    return (predicted == valid_labels).sum().item()


if __name__ == "__main__":
    from os import environ

    environ["CUDA_VISIBLE_DEVICES"] = "3"
    network_train(PruningNetwork([1, 64, 64, 10]))
