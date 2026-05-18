# Задание 2.2: Зигзаг обход матрицы

import numpy as np


def zigzag(matrix: np.ndarray) -> np.ndarray:
    """Зигзаг обход матрицы -> массив"""
    h, w = matrix.shape
    result = []

    for diag_sum in range(h + w - 1):
        diagonal = []
        for i in range(max(0, diag_sum - w + 1), min(h, diag_sum + 1)):
            j = diag_sum - i
            if 0 <= j < w:
                diagonal.append(matrix[i, j])

        if diag_sum % 2 == 0:
            diagonal.reverse()

        result.extend(diagonal)

    return np.array(result)


def inverse_zigzag(array: np.ndarray, h: int, w: int) -> np.ndarray:
    """Обратный зигзаг: массив -> матрица"""
    matrix = np.zeros((h, w), dtype=array.dtype)
    idx = 0

    for diag_sum in range(h + w - 1):
        diagonal_positions = []
        for i in range(max(0, diag_sum - w + 1), min(h, diag_sum + 1)):
            j = diag_sum - i
            if 0 <= j < w:
                diagonal_positions.append((i, j))

        if diag_sum % 2 == 0:
            diagonal_positions.reverse()

        for pos in diagonal_positions:
            matrix[pos] = array[idx]
            idx += 1

    return matrix