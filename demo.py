"""
conda activate test_env
cd Python/TestNetwork
ls
python demo.py
"""
from GateDecorator.Train import train_model, valid_model
from GateDecorator.GatedNet import GatedNet

if __name__ == "__main__":
    import csv

    gated_net = GatedNet(1)
    with open("accuracy.csv", "w") as csv_file:
        writer = csv.writer(csv_file)

        train_model(gated_net, epochs=10, lr=1e-4)
        writer.writerow([0, valid_model(gated_net), 1])

        gated_net.to_gbn()
        for index in range(30):
            gated_net.freeze()
            train_model(gated_net, epochs=10, lr=1e-3)
            gated_net.prune(64)
            writer.writerow([index + 1, valid_model(gated_net), 0])
            gated_net.melt()
            train_model(gated_net, epochs=3, lr=1e-5)
            writer.writerow([index, valid_model(gated_net), 1])
        gated_net.to_bn()
        writer.writerow([-1, valid_model(gated_net), 1])
