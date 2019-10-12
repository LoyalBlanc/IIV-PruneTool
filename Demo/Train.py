import torch
import torch.nn as nn
from torch.optim import Adam
from Demo.Dataset import get_train_loader, get_valid_loader


def train_model(model, epochs=10, batch_size=10000, lr=1e-3):
    model.cuda()
    model.train()

    train_loader = get_train_loader(batch_size)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fuc = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        sum_loss = 0
        for index, (images, labels) in enumerate(train_loader):
            outputs = model(images.cuda())
            loss = loss_fuc(outputs, labels.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        print('Epoch [{}/{}],  Loss: {:.4f}'.format(epoch + 1, epochs, sum_loss))


def valid_model(model, batch_size=10000):
    model.cuda()
    model.eval()

    valid_loader = get_valid_loader(batch_size)
    total = 0
    correct = 0
    with torch.no_grad():
        for index, (images, labels) in enumerate(valid_loader):
            outputs = model(images.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()
    print("Accuracy:{}".format(100 * correct / total))