import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_train_loader(batch_size=100):
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    # 60000:1*32*32
    train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def get_valid_loader(batch_size=100):
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    # 10000:1*32*32
    valid_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    return valid_loader


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
