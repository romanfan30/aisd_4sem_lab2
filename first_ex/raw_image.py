# Задание 1.1: Работа с собственным RAW форматом

import os
import struct
import numpy as np
from PIL import Image

OUT_DIR = "output"


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def p(name):
    return os.path.join(OUT_DIR, name)


def save_to_raw(img: np.ndarray, filename: str, img_type: str, color_space: int = 0):
    """Сохранение изображения в RAW формат"""
    ensure_out()

    height, width = img.shape[:2]

    type_code = {
        'bw': 1,
        'grayscale': 2,
        'color': 3
    }[img_type]

    header = struct.pack('=BII B', type_code, width, height, color_space)
    data = img.astype(np.uint8).tobytes()
    full_path = p(filename)

    with open(full_path, 'wb') as f:
        f.write(header)
        f.write(data)

    return len(header) + len(data)


def load_from_raw(filename: str):
    """Загрузка изображения из RAW формата"""
    full_path = p(filename)

    with open(full_path, 'rb') as f:
        header_size = struct.calcsize('=BII B')
        header = f.read(header_size)

        type_code, width, height, color_space = struct.unpack('=BII B', header)
        data = f.read()

        if type_code == 3:
            img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        else:
            img = np.frombuffer(data, dtype=np.uint8).reshape((height, width))

        return img, type_code, color_space


def to_grayscale(img):
    """Конвертация в оттенки серого"""
    return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def to_bw(img):
    """Черно-белое без dithering"""
    gray = to_grayscale(img)
    return (gray > 127).astype(np.uint8) * 255


def to_bw_dithered(img):
    """Черно-белое с dithering через PIL"""
    if len(img.shape) == 3:
        pil_img = Image.fromarray(img)
    else:
        pil_img = Image.fromarray(img).convert("L")

    bw_pil = pil_img.convert(mode="1")
    return np.array(bw_pil).astype(np.uint8) * 255


def analyze(original_size, raw_sizes, width, height):
    """Анализ размеров файлов"""
    rgb_raw = raw_sizes['RGB']

    print("\n" + "=" * 60)
    print("АНАЛИЗ РАЗМЕРОВ ФАЙЛОВ")
    print("=" * 60)
    print(f"{'Формат':<20} {'Размер (байт)':<15} {'Сжатие':<10} {'Bpp':<10}")
    print("-" * 60)

    print(f"{'Original PNG':<20} {original_size:<15} {'-':<10} {original_size / (width * height):<10.2f}")

    for name, size in raw_sizes.items():
        ratio = original_size / size if size > 0 else 0
        bpp = size / (width * height)
        print(f"{name + ' RAW':<20} {size:<15} {ratio:<10.2f} {bpp:<10.2f}")

    print("=" * 60)