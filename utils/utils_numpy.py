import numpy as np


def list_extend_row_col(matrix, extend_layer_index):
    def get_related_row(mat, row_index):
        return np.where(mat[row_index] == 1)[0]

    def get_related_col(mat, col_index):
        return np.where(mat[:, col_index] == 1)[0]

    def append_extend_row(row_list, col_list, row_index):
        for row in get_related_row(matrix, row_index):
            if row not in row_list:
                row_list.append(row)
                append_extend_col(row_list, col_list, row)

    def append_extend_col(related_row_list, related_col_list, col_index):
        for col in get_related_col(matrix, col_index):
            if col not in related_col_list:
                related_col_list.append(col)
                append_extend_row(related_row_list, related_col_list, col)

    extend_row_list = []
    extend_col_list = []
    append_extend_col(extend_row_list, extend_col_list, extend_layer_index)

    return extend_row_list, extend_col_list


def list_all_extend_row_col(matrix):
    extend_list = []
    matrix_range = matrix.shape[0]
    for extend_layer_index in range(matrix_range):
        extend_list.append(list_extend_row_col(matrix, extend_layer_index))
    return extend_list
