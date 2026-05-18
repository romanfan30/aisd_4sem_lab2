# Задание 2.1: Продвинутое изменение размера изображений с билинейной интерполяцией

import numpy as np


def linear_interp(x1, y1, x2, y2, x):
    """Линейная интерполяция между двумя точками"""
    if x2 == x1:
        return y1

    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def bilinear_interp(x1, y1, x2, y2, z11, z12, z21, z22, x, y):
    """
    Билинейная интерполяция для 4 точек на плоскости
    """
    r1 = linear_interp(x1, z11, x2, z21, x)

    r2 = linear_interp(x1, z12, x2, z22, x)

    return linear_interp(y1, r1, y2, r2, y)


def resize_bilinear(img: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """
    Изменение размера изображения с билинейной интерполяцией
    """
    old_height, old_width = img.shape[:2]
    is_color = len(img.shape) == 3

    if is_color:
        output = np.zeros((new_height, new_width, img.shape[2]), dtype=np.uint8)
    else:
        output = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            if new_height > 1:
                old_i = i * (old_height - 1) / (new_height - 1)
            else:
                old_i = 0

            if new_width > 1:
                old_j = j * (old_width - 1) / (new_width - 1)
            else:
                old_j = 0

            # Найти 4 ближайших пикселя
            i1 = int(np.floor(old_i))
            i2 = min(i1 + 1, old_height - 1)
            j1 = int(np.floor(old_j))
            j2 = min(j1 + 1, old_width - 1)

            # Билинейная интерполяция
            if is_color:
                # Обрабатываем каждый канал отдельно
                for c in range(img.shape[2]):
                    value = bilinear_interp(
                        j1, i1, j2, i2,
                        img[i1, j1, c],  # Верхний левый
                        img[i2, j1, c],  # Нижний левый
                        img[i1, j2, c],  # Верхний правый
                        img[i2, j2, c],  # Нижний правый
                        old_j, old_i
                    )
                    output[i, j, c] = np.clip(value, 0, 255)
            else:
                # Grayscale
                value = bilinear_interp(
                    j1, i1, j2, i2,
                    img[i1, j1],
                    img[i2, j1],
                    img[i1, j2],
                    img[i2, j2],
                    old_j, old_i
                )
                output[i, j] = np.clip(value, 0, 255)

    return output


def downsample_simple(img: np.ndarray, factor: int = 2) -> np.ndarray:
    """Простое уменьшение изображения (децимация)"""
    return img[::factor, ::factor]


def upsample_simple(img: np.ndarray, factor: int = 2) -> np.ndarray:
    """Простое увеличение изображения (дублирование пикселей)"""
    h, w = img.shape[:2]
    is_color = len(img.shape) == 3

    if is_color:
        out = np.zeros((h * factor, w * factor, img.shape[2]), dtype=img.dtype)
    else:
        out = np.zeros((h * factor, w * factor), dtype=img.dtype)

    for i in range(factor):
        for j in range(factor):
            out[i::factor, j::factor] = img

    return out