"""
screen -R cifar
conda activate test_env
cd Python/TestNetwork
ls
python demo.py
"""
from functools import reduce
from operator import mul
from GateDecorator.Train import train_model, valid_model
from GateDecorator.GatedNet import GatedNet


def get_parameters_number(network):
    return sum([reduce(mul, param.size(), 1) for param in network.parameters()])


if __name__ == "__main__":
    import csv
    from os import environ

    environ["CUDA_VISIBLE_DEVICES"] = "1"
    batch_size = 10000
    gated_net = GatedNet(3)
    file_name = "accuracy_CIFAR.csv"
    print(get_parameters_number(gated_net))

    with open(file_name, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Epoch", "Accuracy", "fine", "Param"])

        train_model(gated_net, batch_size=batch_size, epochs=10, lr=1e-4)
        accuracy = valid_model(gated_net)
        parameter = get_parameters_number(gated_net)
        print(accuracy, parameter)
        writer.writerow([0, accuracy, 1, parameter])

    gated_net.to_gbn()
    for index in range(30):
        gated_net.freeze()
        train_model(gated_net, batch_size=batch_size, epochs=10, lr=1e-3)
        gated_net.prune(64)
        accuracy1 = valid_model(gated_net)
        parameter = get_parameters_number(gated_net)

        gated_net.melt()
        train_model(gated_net, batch_size=batch_size, epochs=3, lr=1e-5)
        accuracy2 = valid_model(gated_net)
        print(parameter, accuracy1, accuracy2)
        with open(file_name, "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([index + 1, accuracy2, accuracy1, parameter])

    gated_net.to_bn()
    accuracy = valid_model(gated_net)
    parameter = get_parameters_number(gated_net)
    print(accuracy, parameter)
    with open(file_name, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([-1, accuracy, 1, parameter])
