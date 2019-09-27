import torch
import torch.nn as nn
from torch.optim import Adam
from GateDecorator.Dataset import get_train_loader


def train_model(model, epochs=10):
    model.cuda()
    model.train()

    train_loader = get_train_loader()
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fuc = nn.CrossEntropyLoss()

    total_step = len(train_loader)
    for epoch in range(epochs):
        sum_loss = 0
        for index, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)

            loss = loss_fuc(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch, epochs, index, total_step, sum_loss))
