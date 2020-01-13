import numpy as np
import torch
import torch.nn as nn


def get_related_row(matrix, related_layer_index):
    return np.where(matrix[related_layer_index] == 1)[0]


def get_related_col(matrix, related_layer_index):
    return np.where(matrix[:, related_layer_index] == 1)[0]


def get_extend_row_col(matrix, related_layer_index):
    def append_extend_row(related_row_list, related_col_list, row_index):
        for link in get_related_row(matrix, row_index):
            if link not in related_row_list:
                related_row_list.append(link)
                append_extend_col(related_row_list, related_col_list, link)

    def append_extend_col(related_row_list, related_col_list, col_index):
        for link in get_related_col(matrix, col_index):
            if link not in related_col_list:
                related_col_list.append(link)
                append_extend_row(related_row_list, related_col_list, link)

    row_list = []
    col_list = []
    append_extend_col(row_list, col_list, related_layer_index)

    return row_list, col_list


def get_model_flops(network, data):
    flops = []

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
                net.register_forward_hook(conv_hook)
            if isinstance(net, nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, nn.ReLU):
                net.register_forward_hook(relu_hook)
            return
        for child in children:
            register(child)

    register(network)
    _ = network(data)
    return sum(flops)


if __name__ == "__main__":
    skip_mat = np.tril(np.array([[0, 0, 0, 0],
                                 [1, 0, 0, 0],
                                 [1, 1, 0, 0],
                                 [1, 0, 1, 0]]), k=-1)
    test_row_list, test_col_list = get_extend_row_col(skip_mat, 1)
    assert list(set(test_row_list)) == [0, 1, 2] and list(set(test_col_list)) == [1, 2, 3]

    print("Pass the unit exam!")
