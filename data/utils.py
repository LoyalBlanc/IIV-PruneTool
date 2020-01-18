import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def save_param(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_param(model, save_path):
    model.load_state_dict(torch.load(save_path), strict=True)


def get_train_loader(batch_size=1000):
    transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def get_valid_loader(batch_size=1000):
    transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    valid_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    return valid_loader


def valid_model(model, batch_size=1000):
    model.cuda()
    model.eval()
    valid_loader = get_valid_loader(batch_size)
    total = 0
    correct = 0
    with torch.no_grad():
        for index, (images, labels) in enumerate(valid_loader):
            outputs = model(torch.cat((images, images, images), dim=1).cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()
    print("Accuracy:{}".format(100 * correct / total))
