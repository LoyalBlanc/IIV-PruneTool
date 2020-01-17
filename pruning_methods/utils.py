import torch.nn as nn
# from .abstract_method import get_the_module_name


def get_model_flops(model, data):
    model.eval()
    flops = []
    child_hook = []

    def conv_hook(self, input_tensor, output_tensor):
        batch_size, _, _, _ = input_tensor[0].size()
        output_channels, output_height, output_width = output_tensor[0].size()
        flops.append(batch_size * output_channels * output_height * output_width)

    def bn_hook(self, input_tensor, output_tensor):
        flops.append(input_tensor[0].nelement())

    def relu_hook(self, input_tensor, output_tensor):
        flops.append(input_tensor[0].nelement())

    def register(net):
        children = list(net.children())
        if not children:
            if isinstance(net, nn.Conv2d):
                child_hook.append(net.register_forward_hook(conv_hook))
            if isinstance(net, nn.BatchNorm2d):
                child_hook.append(net.register_forward_hook(bn_hook))
            if isinstance(net, nn.ReLU):
                child_hook.append(net.register_forward_hook(relu_hook))
            return
        for child in children:
            register(child)

    register(model)
    _ = model(data)
    for hook in child_hook:
        hook.remove()
    return sum(flops)

