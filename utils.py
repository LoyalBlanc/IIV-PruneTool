import torch
import torch.optim as op
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def save_param(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_param(model, save_path):
    model.load_state_dict(torch.load(save_path), strict=True)


def get_train_loader(batch_size=1000):
    transform = transforms.Compose([transforms.Resize(96), transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def get_valid_loader(batch_size=1000):
    transform = transforms.Compose([transforms.Resize(96), transforms.ToTensor()])
    valid_dataset = torchvision.datasets.CIFAR10(root='data', train=False, transform=transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    return valid_loader


def train_one_epoch(model, dataset, criterion, lr):
    optimizer = op.Adam(model.parameters(), lr=lr)
    step_loss = 0
    for index, (images, labels) in enumerate(dataset):
        # images = torch.cat((images, images, images), dim=1)
        outputs = model(images.cuda())
        loss = criterion(outputs, labels.cuda()) + 1e-3 * model.regularization
        model.regularization = 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss += loss.item()
    return step_loss, optimizer.param_groups[0]['lr']


def valid_model(model, batch_size=1000, verbose=True):
    model.cuda()
    model.eval()
    valid_loader = get_valid_loader(batch_size)
    total = 0
    correct = 0
    with torch.no_grad():
        for index, (images, labels) in enumerate(valid_loader):
            # images = torch.cat((images, images, images), dim=1)
            outputs = model(images.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()
    acc = 100 * correct / total
    if verbose:
        print("Accuracy:{}".format(acc))
    return acc
