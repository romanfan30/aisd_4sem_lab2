# 2.5 Кодирование алгоритмом Хаффмана

import math

def bits_from_counts(counts):
    return [int(x) for x in counts]

def make_huffman_table(bits, values):
    code = 0
    k = 0
    table = {}
    decode_table = {}

    for bit_length, count in enumerate(bits, start=1):
        for _ in range(count):
            symbol = values[k]
            bit_string = format(code, f'0{bit_length}b')
            table[symbol] = bit_string
            decode_table[bit_string] = symbol
            code += 1
            k += 1
        code <<= 1

    return table, decode_table

STD_LUMINANCE_DC_BITS = bits_from_counts(
    [0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0]
)

STD_LUMINANCE_DC_VALUES = list(range(12))

STD_LUMINANCE_AC_BITS = bits_from_counts(
    [0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,125]
)

STD_LUMINANCE_AC_VALUES = [
0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,
0x41,0x06,0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,
0x81,0x91,0xA1,0x08,0x23,0x42,0xB1,0xC1,0x15,0x52,
0xD1,0xF0,0x24,0x33,0x62,0x72,0x82,0x09,0x0A,0x16,
0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,0x29,0x2A,
0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,
0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,0x56,0x57,
0x58,0x59,0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,
0x6A,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,0x83,
0x84,0x85,0x86,0x87,0x88,0x89,0x8A,0x92,0x93,0x94,
0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,
0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6,
0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,
0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,
0xD9,0xDA,0xE1,0xE2,0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,
0xE9,0xEA,0xF1,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,
0xF9,0xFA
]
# Генерация таблиц
DC_HUFFMAN_TABLE, DC_HUFFMAN_DECODE = make_huffman_table(
    STD_LUMINANCE_DC_BITS,
    STD_LUMINANCE_DC_VALUES
)

AC_HUFFMAN_TABLE_RAW, AC_HUFFMAN_DECODE_RAW = make_huffman_table(
    STD_LUMINANCE_AC_BITS,
    STD_LUMINANCE_AC_VALUES
)

# Преобразуем AC символы из 0xRS в (run, size)
AC_HUFFMAN_TABLE = {}
AC_HUFFMAN_DECODE = {}

for symbol, code in AC_HUFFMAN_TABLE_RAW.items():
    run = (symbol >> 4) & 0x0F
    size = symbol & 0x0F
    AC_HUFFMAN_TABLE[(run, size)] = code
    AC_HUFFMAN_DECODE[code] = (run, size)

# --- ФУНКЦИИ КОДИРОВАНИЯ ---

def huffman_encode_dc_block(category, amplitude):
    huff_code = DC_HUFFMAN_TABLE[category]
    if category == 0:
        return huff_code
    amp_bits = format(amplitude, f'0{category}b')
    return huff_code + amp_bits

def huffman_encode_ac_block(run, category, amplitude):
    if (run, category) not in AC_HUFFMAN_TABLE:
        raise ValueError(f"Huffman код не найден для пары (run={run}, category={category})")
    huff_code = AC_HUFFMAN_TABLE[(run, category)]
    if category == 0:
        return huff_code
    amp_bits = format(amplitude, f'0{category}b')
    return huff_code + amp_bits

def huffman_decode_dc_block(bitstream, pos=0):
    code = ''
    i = pos
    while i < len(bitstream):
        code += bitstream[i]
        i += 1
        if code in DC_HUFFMAN_DECODE:
            break
    category = DC_HUFFMAN_DECODE[code]
    if category > 0:
        amp_bits = bitstream[i:i+category]
        i += category
        amplitude = int(amp_bits,2)
    else:
        amplitude = 0
    return category, amplitude, i

def huffman_decode_ac_block(bitstream, pos=0):
    code = ''
    i = pos
    while i < len(bitstream):
        code += bitstream[i]
        i += 1
        if code in AC_HUFFMAN_DECODE:
            break
    run, category = AC_HUFFMAN_DECODE[code]
    if category > 0:
        amp_bits = bitstream[i:i+category]
        i += category
        amplitude = int(amp_bits,2)
    else:
        amplitude = 0
    return run, category, amplitude, i