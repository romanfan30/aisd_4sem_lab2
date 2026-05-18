# Задание 2.3: Разностное кодирование DC и RLE для AC

import numpy as np


def dc_differential_encoding(dc_coeffs):
    """Разностное кодирование DC коэффициентов"""
    dc_coeffs = np.array(dc_coeffs, dtype=int)
    if len(dc_coeffs) == 0:
        return []

    diff_codes = np.diff(dc_coeffs, prepend=0)
    diff_codes[0] = dc_coeffs[0]
    return diff_codes.tolist()


def dc_differential_decoding(diff_codes):
    """Декодирование DC коэффициентов"""
    dc_coeffs = np.array(diff_codes, dtype=int)
    if len(dc_coeffs) == 0:
        return np.array([])

    result = np.zeros_like(dc_coeffs)
    result[0] = dc_coeffs[0]
    for i in range(1, len(dc_coeffs)):
        result[i] = result[i - 1] + dc_coeffs[i]

    return result


def rle_encode_ac(ac_coeffs):
    """RLE кодирование AC коэффициентов (63 элемента)"""
    if len(ac_coeffs) != 63:
        raise ValueError("Длина AC массива должна быть 63")

    rle_result = []
    zero_run = 0

    last_nonzero_idx = -1
    for i in range(62, -1, -1):
        if ac_coeffs[i] != 0:
            last_nonzero_idx = i
            break

    if last_nonzero_idx == -1:
        return [(0, 0)]

    for i in range(last_nonzero_idx + 1):
        val = ac_coeffs[i]
        if val == 0:
            zero_run += 1
            if zero_run == 16:
                rle_result.append((15, 0))
                zero_run = 0
        else:
            rle_result.append((zero_run, val))
            zero_run = 0

    if last_nonzero_idx < 62:
        rle_result.append((0, 0))

    return rle_result


def rle_decode_ac(rle_codes, length=63):
    """Декодирование RLE AC коэффициентов"""
    ac_coeffs = []

    for run, value in rle_codes:
        if run == 0 and value == 0:
            break
        ac_coeffs.extend([0] * run)
        ac_coeffs.append(value)

    ac_coeffs.extend([0] * (length - len(ac_coeffs)))
    return np.array(ac_coeffs[:length])