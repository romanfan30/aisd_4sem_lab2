# Задание 2.4: Кодирование переменной длины (VLC)

import numpy as np


def get_category(value):
    """Определить категорию числа"""
    if value == 0:
        return 0
    return int(np.floor(np.log2(abs(value)))) + 1


def encode_amplitude(value, category):
    """Кодировать амплитуду"""
    if value == 0:
        return 0
    if value > 0:
        return value
    return (1 << category) - 1 + value


def decode_amplitude(amplitude, category):
    """Декодировать амплитуду"""
    if category == 0:
        return 0
    threshold = 1 << (category - 1)
    if amplitude >= threshold:
        return amplitude
    return amplitude - (1 << category) + 1


def vlc_encode_value(value):
    """VLC кодирование одного значения"""
    category = get_category(value)
    if category == 0:
        return (0, 0)

    if value > 0:
        amplitude = value
    else:
        amplitude = (1 << category) - 1 + value

    return (category, amplitude)


def vlc_decode_value(category, amplitude):
    """VLC декодирование"""
    return decode_amplitude(amplitude, category)


def vlc_rle_encode_ac(ac_coeffs):
    """Объединенное RLE + VLC для AC"""
    if len(ac_coeffs) != 63:
        raise ValueError("AC массив должен содержать 63 элемента")

    vlc_rle_codes = []
    zero_run = 0

    last_nonzero = -1
    for i in range(62, -1, -1):
        if ac_coeffs[i] != 0:
            last_nonzero = i
            break

    if last_nonzero == -1:
        return [(0, 0, 0)]

    for i in range(last_nonzero + 1):
        value = ac_coeffs[i]
        if value == 0:
            zero_run += 1
            if zero_run == 16:
                vlc_rle_codes.append((15, 0, 0))
                zero_run = 0
        else:
            category = get_category(value)
            amplitude = encode_amplitude(value, category)
            vlc_rle_codes.append((zero_run, category, amplitude))
            zero_run = 0

    vlc_rle_codes.append((0, 0, 0))
    return vlc_rle_codes


def vlc_rle_decode_ac(vlc_rle_codes, length=63):
    ac_coeffs = []

    for run, category, amplitude in vlc_rle_codes:

        if run == 0 and category == 0:
            break

        if run == 15 and category == 0:
            ac_coeffs.extend([0] * 16)
            continue

        ac_coeffs.extend([0] * run)

        value = decode_amplitude(amplitude, category)
        ac_coeffs.append(value)

    ac_coeffs.extend([0] * (length - len(ac_coeffs)))
    return np.array(ac_coeffs[:length])