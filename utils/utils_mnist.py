import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader


def get_train_loader(batch_size=100):
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    # 60000:32*32
    train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def get_valid_loader(batch_size=100):
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    # 10000:32*32
    valid_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    return valid_loader


def save_param(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_param(model, save_path):
    model.load_state_dict(torch.load(save_path), strict=True)


def train_model(model, epochs=10, batch_size=10000, lr=1e-3, regular=False):
    model.cuda()
    model.train()
    train_loader = get_train_loader(batch_size)
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        sum_loss = 0
        for index, (images, labels) in enumerate(train_loader):
            outputs = model(images.cuda())
            loss = nn.CrossEntropyLoss()(outputs, labels.cuda())
            if regular:
                loss += 1e-3 * model.regularization
                model.regularization = 0
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
