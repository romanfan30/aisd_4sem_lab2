# Задание 1.2: Преобразование цветовых пространств RGB <-> YCbCr

import numpy as np


def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    """RGB -> YCbCr преобразование"""

    if img.dtype == np.bool_:
        img = img.astype(np.uint8) * 255

    elif img.max() <= 1 and img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)

    img = img.astype(np.float32)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

    out = np.stack([y, cb, cr], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def ycbcr_to_rgb(img: np.ndarray) -> np.ndarray:
    """YCbCr -> RGB преобразование"""
    img = img.astype(np.float32)

    y = img[:, :, 0]
    cb = img[:, :, 1] - 128
    cr = img[:, :, 2] - 128

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    out = np.stack([r, g, b], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Конвертация в оттенки серого"""

    if img.dtype == np.bool_:
        img = img.astype(np.uint8) * 255
    elif img.max() <= 1:
        img = (img * 255).astype(np.uint8)

    return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)