import torch
import torchvision.models as models
import network_analyzer as nwa

if __name__ == "__main__":
    model = models.resnet18()
    nwa.analyze_network(model, torch.ones(1, 3, 224, 224), verbose=False, for_pruning=True)
