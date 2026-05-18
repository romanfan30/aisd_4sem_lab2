# Задание 2.6: Управление качеством сжатия

import numpy as np
import math


def calculate_scaling_factor(quality):
    """Вычислить коэффициент масштабирования S"""
    quality = np.clip(quality, 1, 100)
    if quality < 50:
        return 5000 / quality
    return 200 - 2 * quality


def adapt_quantization_table(base_table, quality):
    """
    Адаптация таблицы квантования по качеству
    Использует формулу из стандарта JPEG
    """
    quality = max(1, min(quality, 100))

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    # Создаем новую таблицу
    result = np.zeros_like(base_table, dtype=np.float32)

    for i in range(base_table.shape[0]):
        for j in range(base_table.shape[1]):
            # ВАЖНО: используем CEIL, как в стандарте JPEG
            scaled = math.ceil((base_table[i, j] * scale) / 100.0)
            # Ограничиваем диапазон [1, 255]
            scaled = min(max(int(scaled), 1), 255)
            result[i, j] = float(scaled)

    return result