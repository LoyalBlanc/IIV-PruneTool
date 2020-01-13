from PruneMethod.minimum_weight import find_minimum_weight
from utils import get_model_flops


def prune_network(network, example_data, method, prune_ratio):
    flops_before = get_model_flops(network, example_data)
    flops_target = int(flops_before * (1 - prune_ratio))
    print("FLOPs before pruning is %d, target is %d." % (flops_before, flops_target))

    flops_now = flops_before
    try:
        while flops_now > flops_target:
            if method == "minimum_weight":
                layer_index, channel_index, minimum_weight = find_minimum_weight(network)
                network.prune(layer_index, channel_index)
                flops_now = get_model_flops(network, example_data)
                print("Prune Channel %d in Layer %d (Weight %.4f)." % (layer_index, channel_index, minimum_weight))

    except RuntimeError:
        print("Fail to prune the network, one layer lost all the channels.")
    except Exception as e:
        print("Fatal Error!\n %s" % e)
    else:
        print("Successfully prune the network, the FLOPs now is %d" % flops_now)


if __name__ == "__main__":
    pass
