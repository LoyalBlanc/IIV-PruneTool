import torch


def save_param(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_param(model, save_path):
    model.load_state_dict(torch.load(save_path), strict=True)

