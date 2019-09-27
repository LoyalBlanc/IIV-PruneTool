import torch
import torchvision
import torchvision.transforms as transforms


def get_train_loader():
    transform = transforms.Compose([transforms.ToTensor()])
    # 60000:28*28
    train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    return train_loader


def get_valid_loader():
    transform = transforms.Compose([transforms.ToTensor()])
    # 10000:28*28
    valid_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transform)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=8, shuffle=False)
    return valid_loader

