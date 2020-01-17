import torch

import types
from pruning_tools.network_analyzer import network_detector
from .support_modules.__init__ import *


def analyze_network(network, input_data, verbose=False, for_pruning=True):
    network.cuda()
    unit_dict = network_detector.detect_network(network, input_data.cuda())
    if verbose:
        print(network.__repr__)
        print(network_detector.get_unit_name_dict_info(unit_dict))
    if for_pruning:
        network.unit_dict = unit_dict
        network.pruning_range = []
        network.hook = None
        network.regularization = 0
        for unit_name in unit_dict.keys():
            unit = eval("network" + unit_name)
            if unit.is_layer:
                _layer_change(unit)
            else:
                _non_layer_change(unit)
        network.prune_spec_channel = types.MethodType(_network_prune_spec_channel, network)
        network.cpu()


# Todo:
#  After adding supported modules, the function of
#  deciding to select which module need to be determined

def _layer_change(unit):
    # Conv2d
    module = conv2d
    unit.prune_ipc = types.MethodType(module.layer_prune_ipc, unit)
    unit.prune_opc = types.MethodType(module.layer_prune_opc, unit)


def _non_layer_change(unit):
    # BatchNorm2d
    module = batchnorm2d
    unit.prune_ipc = types.MethodType(module.non_layer_prune, unit)
    if unit.track_running_stats:
        unit.track_running_stats = False
        unit.running_mean = None
        unit.running_var = None
        unit.num_batches_tracked = None


def _network_prune_spec_channel(self, layer_name, channel):
    layer_info = 'Layer '
    spec_layer = self.unit_dict[layer_name]
    layer_affect_opc, layer_affect_ipc = spec_layer.affect_opc, spec_layer.affect_ipc
    for sub_layer_name in layer_affect_opc:
        layer_info += "%s, " % sub_layer_name
        eval('self' + sub_layer_name).prune_opc(channel)
        for sub_layer in self.unit_dict[sub_layer_name].next:
            eval('self' + sub_layer).prune_ipc(channel)
    for sub_layer_name in layer_affect_ipc:
        eval('self' + sub_layer_name).prune_ipc(channel)
    return layer_info


if __name__ == "__main__":
    import torchvision.models as models

    model = models.resnet18()
    analyze_network(model, torch.ones(1, 3, 224, 224))
    for test_unit_name in model.unit_dict:
        model_backup = models.resnet18()
        analyze_network(model_backup, torch.ones(1, 3, 64, 64))
        test_unit = eval("model_backup" + test_unit_name)
        if test_unit.is_layer:
            model_backup.prune_spec_channel(test_unit_name, 0)
            try:
                _ = model_backup(torch.ones(1, 3, 224, 224))
            except Exception as e:
                print(test_unit_name, e)
