# Задание 1.3: Downsampling/Upsampling/Resizing

import numpy as np


def downsample(img: np.ndarray) -> np.ndarray:
    """Децимация с коэффициентом 2"""
    return img[::2, ::2]


def upsample(img: np.ndarray) -> np.ndarray:
    """Upsampling с дублированием пикселей"""
    h, w = img.shape[:2]

    if len(img.shape) == 3:
        out = np.zeros((h * 2, w * 2, img.shape[2]), dtype=img.dtype)
    else:
        out = np.zeros((h * 2, w * 2), dtype=img.dtype)

    out[::2, ::2] = img
    out[1::2, ::2] = img
    out[::2, 1::2] = img
    out[1::2, 1::2] = img

    return out


def linear_interp(x1, y1, x2, y2, x):
    if x2 == x1:
        return float(y1)

    y1 = float(y1)
    y2 = float(y2)

    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def linear_spline(xs, ys, x):
    """Вычисление линейного сплайна"""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]

    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            return linear_interp(xs[i], ys[i], xs[i + 1], ys[i + 1], x)

    return ys[-1]


def bilinear_interp(x1, y1, x2, y2, z11, z12, z21, z22, x, y):
    """Билинейная интерполяция для четырех точек"""
    r1 = linear_interp(x1, z11, x2, z21, x)
    r2 = linear_interp(x1, z12, x2, z22, x)
    return linear_interp(y1, r1, y2, r2, y)


def resize_bilinear(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Изменение размера изображения с билинейной интерполяцией"""
    old_h, old_w = img.shape[:2]
    is_color = len(img.shape) == 3

    if is_color:
        out = np.zeros((new_h, new_w, img.shape[2]), dtype=np.uint8)
    else:
        out = np.zeros((new_h, new_w), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            old_i = i * (old_h - 1) / (new_h - 1) if new_h > 1 else 0
            old_j = j * (old_w - 1) / (new_w - 1) if new_w > 1 else 0

            i1 = int(np.floor(old_i))
            i2 = min(i1 + 1, old_h - 1)
            j1 = int(np.floor(old_j))
            j2 = min(j1 + 1, old_w - 1)

            if is_color:
                for c in range(img.shape[2]):
                    val = bilinear_interp(
                        j1, i1, j2, i2,
                        img[i1, j1, c], img[i2, j1, c],
                        img[i1, j2, c], img[i2, j2, c],
                        old_j, old_i
                    )
                    out[i, j, c] = np.clip(val, 0, 255)
            else:
                val = bilinear_interp(
                    j1, i1, j2, i2,
                    img[i1, j1], img[i2, j1],
                    img[i1, j2], img[i2, j2],
                    old_j, old_i
                )
                out[i, j] = np.clip(val, 0, 255)

    return out