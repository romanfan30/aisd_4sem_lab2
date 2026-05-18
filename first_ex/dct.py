# Задание 1.3: Дискретное косинусное преобразование

import numpy as np

# Стандартная таблица квантования JPEG
JPEG_QUANT_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)


def split_blocks(img: np.ndarray, block_size=8):
    """Разбиение изображения на блоки"""
    h, w = img.shape[:2]
    blocks = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i + block_size, j:j + block_size]

            # Дополнение до нужного размера
            if block.shape[0] < block_size or block.shape[1] < block_size:
                padded = np.zeros((block_size, block_size), dtype=img.dtype)
                padded[:block.shape[0], :block.shape[1]] = block
                block = padded

            blocks.append(block)

    return blocks


def merge_blocks(blocks, img_shape, block_size=8):
    """Сборка изображения из блоков"""
    h, w = img_shape[:2]
    img = np.zeros(img_shape, dtype=blocks[0].dtype)

    block_idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = blocks[block_idx]
            h_end = min(i + block_size, h)
            w_end = min(j + block_size, w)
            img[i:h_end, j:w_end] = block[:h_end - i, :w_end - j]
            block_idx += 1

    return img


def dct_matrix(N):
    """Матрица ДКП размера NxN"""
    C = np.zeros((N, N), dtype=np.float32)
    for k in range(N):
        for n in range(N):
            if k == 0:
                C[k, n] = np.sqrt(1 / N)
            else:
                C[k, n] = np.sqrt(2 / N) * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    return C


def dct_2d_matrix(block: np.ndarray) -> np.ndarray:
    """Прямое ДКП через матричное умножение"""
    N = block.shape[0]
    C = dct_matrix(N)
    return C @ block @ C.T


def idct_2d_matrix(dct_block: np.ndarray) -> np.ndarray:
    N = dct_block.shape[0]
    C = dct_matrix(N)
    return C.T @ dct_block @ C


def quantize(dct_coeffs: np.ndarray, quant_matrix: np.ndarray) -> np.ndarray:
    return np.round(dct_coeffs / quant_matrix).astype(np.int16)


def dequantize(quantized: np.ndarray, quant_matrix: np.ndarray) -> np.ndarray:
    return quantized.astype(np.float32) * quant_matrix