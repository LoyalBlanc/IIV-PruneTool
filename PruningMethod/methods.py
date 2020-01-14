import PruningMethod.minimum_weight as mw
import utils


def one_shot_prune(network, example_data, method, prune_ratio):
    flops_before = utils.get_model_flops(network, example_data)
    flops_target = int(flops_before * (1 - prune_ratio))
    print("FLOPs before pruning is %d, target is %d." % (flops_before, flops_target))

    flops_now = flops_before
    try:
        while flops_now > flops_target:
            if method == "minimum_weight":
                mw.prune_one_channel(network)
            else:
                print("No such pruning method.")
            flops_now = utils.get_model_flops(network, example_data)
    except RuntimeError:
        print("Fail to prune the network, one layer lost all the channels.")
    except Exception as e:
        print("Fatal Error!\n %s" % e)
    else:
        print("Successfully prune the network, the FLOPs now is %d" % flops_now)


if __name__ == "__main__":
    pass
