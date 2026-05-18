# JPEG-подобный компрессор

import os
import numpy as np
from PIL import Image

from first_ex.color_space import rgb_to_ycbcr, ycbcr_to_rgb
from first_ex.dct import (split_blocks, merge_blocks, dct_2d_matrix,
                          idct_2d_matrix, quantize, dequantize, JPEG_QUANT_MATRIX)
from first_ex.sampling import downsample, upsample

from second_ex.resize_advanced import resize_bilinear
from second_ex.zigzag import zigzag, inverse_zigzag
from second_ex.DC_rle import dc_differential_encoding, dc_differential_decoding
from second_ex.VLC import (vlc_rle_encode_ac, vlc_rle_decode_ac, vlc_encode_value,
                           get_category, encode_amplitude, decode_amplitude)
from second_ex.huffman_coding import (huffman_encode_dc_block, huffman_encode_ac_block,
                                      huffman_decode_dc_block, huffman_decode_ac_block)
from second_ex.quality_control import adapt_quantization_table
from second_ex.jpeg_file_io import write_compressed_file, read_compressed_file

# Создание директории output
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_output_path(filename):
    """Получить путь в директории output"""
    return os.path.join(OUTPUT_DIR, filename)

def compress_channel(channel, quant_table):
    blocks = split_blocks(channel, block_size=8)

    bitstream = ''
    prev_dc = 0

    for block in blocks:
        block_centered = block.astype(np.float32) - 128
        dct_coeffs = dct_2d_matrix(block_centered)
        quantized = quantize(dct_coeffs, quant_table)
        zigzag_coeffs = zigzag(quantized)

        # DC
        dc = int(zigzag_coeffs[0])
        dc_diff = dc - prev_dc
        prev_dc = dc

        category, amplitude = vlc_encode_value(dc_diff)
        bitstream += huffman_encode_dc_block(category, amplitude)

        # AC
        ac = zigzag_coeffs[1:].astype(np.int16).tolist()
        vlc_rle_ac = vlc_rle_encode_ac(ac)

        for run, category, amplitude in vlc_rle_ac:
            bitstream += huffman_encode_ac_block(run, category, amplitude)

    return bitstream, len(blocks)
def decompress_channel(bitstream, num_blocks, height, width, quant_table):
    blocks = []
    pos = 0
    prev_dc = 0

    for block_idx in range(num_blocks):

        # ---- DC ----
        category, amplitude, pos = huffman_decode_dc_block(bitstream, pos)
        dc_diff = decode_amplitude(amplitude, category)
        dc = prev_dc + dc_diff
        prev_dc = dc

        # ---- AC ----
        vlc_rle_ac = []

        while True:
            run, category, amplitude, pos = huffman_decode_ac_block(bitstream, pos)

            # EOB
            if run == 0 and category == 0:
                vlc_rle_ac.append((0, 0, 0))
                break

            # ZRL
            if run == 15 and category == 0:
                vlc_rle_ac.append((15, 0, 0))
                continue

            # ✅ ВАЖНО: НЕ декодируем amplitude здесь
            vlc_rle_ac.append((run, category, amplitude))

        # ✅ Только тут выполняется decode_amplitude
        ac_coeffs = vlc_rle_decode_ac(vlc_rle_ac)

        # ---- сборка блока ----
        zigzag_coeffs = np.concatenate([[dc], ac_coeffs])
        quantized = inverse_zigzag(zigzag_coeffs, 8, 8)
        dct_coeffs = dequantize(quantized, quant_table)
        block_centered = idct_2d_matrix(dct_coeffs)
        block = np.clip(block_centered + 128, 0, 255).astype(np.uint8)

        blocks.append(block)

    return merge_blocks(blocks, (height, width), 8)

def compress_image(img_path, output_path, quality=75, use_subsampling=True):
    """Сжатие изображения"""
    img = np.array(Image.open(img_path))

    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)

    # RGB -> YCbCr
    ycbcr = rgb_to_ycbcr(img)
    y_channel = ycbcr[:, :, 0]
    cb_channel = ycbcr[:, :, 1]
    cr_channel = ycbcr[:, :, 2]

    # Subsampling для Cb и Cr
    if use_subsampling:
        cb_down = downsample(cb_channel)
        cr_down = downsample(cr_channel)
    else:
        cb_down = cb_channel
        cr_down = cr_channel

    # Адаптация таблицы квантования
    quant_y = adapt_quantization_table(JPEG_QUANT_MATRIX, quality)
    quant_c = adapt_quantization_table(JPEG_QUANT_MATRIX * 1.5, quality)

    # Сжатие каналов
    y_bitstream, y_blocks = compress_channel(y_channel, quant_y)
    cb_bitstream, cb_blocks = compress_channel(cb_down, quant_c)
    cr_bitstream, cr_blocks = compress_channel(cr_down, quant_c)

    # Запись в файл
    image_data = {
        'width': img.shape[1],
        'height': img.shape[0],
        'colorspace': 1,  # YCbCr
        'quality': quality,
        'quant_table': quant_y,
        'quant_table_c': quant_c,
        'num_blocks': y_blocks,
        'cb_blocks': cb_blocks,
        'cr_blocks': cr_blocks,
        'y_bitstream_len': len(y_bitstream),
        'cb_bitstream_len': len(cb_bitstream),
        'bitstream': y_bitstream + cb_bitstream + cr_bitstream,
        'subsampling': use_subsampling
    }

    file_size = write_compressed_file(output_path, image_data)

    print(f"Сжатие завершено:")
    print(f"  Quality: {quality}")
    print(f"  Размер: {file_size} байт")
    print(f"  Файл: {output_path}")

    return file_size


def decompress_image(compressed_path, output_path):
    """Декомпрессия изображения"""
    data = read_compressed_file(compressed_path)

    width = data['width']
    height = data['height']
    quant_y = data['quant_table']
    quant_c = data.get('quant_table_c', quant_y)

    bitstream = data['bitstream']
    y_blocks = data['num_blocks']
    cb_blocks = data.get('cb_blocks', y_blocks // 4)
    cr_blocks = data.get('cr_blocks', y_blocks // 4)

    use_subsampling = data.get('subsampling', True)

    # Разделение битстрима
    y_len = data.get('y_bitstream_len', 0)
    cb_len = data.get('cb_bitstream_len', 0)

    y_bitstream = bitstream[:y_len]
    cb_bitstream = bitstream[y_len:y_len + cb_len]
    cr_bitstream = bitstream[y_len + cb_len:]

    # Декомпрессия Y канала
    y_channel = decompress_channel(y_bitstream, y_blocks, height, width, quant_y)

    # Декомпрессия Cb и Cr каналов
    if use_subsampling:
        cb_height = height // 2
        cb_width = width // 2
    else:
        cb_height = height
        cb_width = width

    cb_channel = decompress_channel(cb_bitstream, cb_blocks, cb_height, cb_width, quant_c)
    cr_channel = decompress_channel(cr_bitstream, cr_blocks, cb_height, cb_width, quant_c)

    # Upsampling для Cb и Cr
    if use_subsampling:
        cb_channel = upsample(cb_channel)
        cr_channel = upsample(cr_channel)

        # Приведение к нужному размеру
        if cb_channel.shape[0] != height or cb_channel.shape[1] != width:
            cb_channel = resize_bilinear(cb_channel, height, width)
        if cr_channel.shape[0] != height or cr_channel.shape[1] != width:
            cr_channel = resize_bilinear(cr_channel, height, width)

    # Сборка YCbCr изображения
    ycbcr = np.stack([y_channel, cb_channel, cr_channel], axis=2)

    # YCbCr -> RGB
    rgb = ycbcr_to_rgb(ycbcr)

    # Сохранение
    img = Image.fromarray(rgb)
    img.save(output_path)

    print(f"Декомпрессия завершена:")
    print(f"  Файл: {output_path}")

    return rgb


def resize_image(img_path, output_path, new_height, new_width):
    """Изменение размера изображения с билинейной интерполяцией"""
    img = np.array(Image.open(img_path))
    resized = resize_bilinear(img, new_height, new_width)

    output_full_path = get_output_path(output_path)
    Image.fromarray(resized).save(output_full_path)

    print(f"Изменение размера завершено:")
    print(f"  Новый размер: {new_height}x{new_width}")
    print(f"  Файл: {output_full_path}")

    return resized