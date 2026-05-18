import struct
import numpy as np
from pathlib import Path

MAGIC_NUMBER = b'MJPG'
VERSION = 2


def write_compressed_file(filename, image_data):
    with open(filename, 'wb') as f:
        f.write(MAGIC_NUMBER)
        f.write(struct.pack('B', VERSION))

        f.write(struct.pack('I', image_data['width']))
        f.write(struct.pack('I', image_data['height']))
        f.write(struct.pack('B', image_data['colorspace']))
        f.write(struct.pack('B', image_data['quality']))
        f.write(struct.pack('B', int(image_data.get('subsampling', True))))

        # Таблица Y
        for val in image_data['quant_table'].flatten():
            f.write(struct.pack('f', val))

        # Таблица C
        for val in image_data['quant_table_c'].flatten():
            f.write(struct.pack('f', val))

        f.write(struct.pack('I', image_data['num_blocks']))
        f.write(struct.pack('I', image_data['cb_blocks']))
        f.write(struct.pack('I', image_data['cr_blocks']))

        f.write(struct.pack('I', image_data['y_bitstream_len']))
        f.write(struct.pack('I', image_data['cb_bitstream_len']))

        bitstream = image_data['bitstream']

        padding = (8 - len(bitstream) % 8) % 8
        bitstream_padded = bitstream + '0' * padding

        byte_data = bytearray()
        for i in range(0, len(bitstream_padded), 8):
            byte_data.append(int(bitstream_padded[i:i + 8], 2))

        f.write(struct.pack('I', len(bitstream)))
        f.write(byte_data)

    return Path(filename).stat().st_size


def read_compressed_file(filename):
    with open(filename, 'rb') as f:
        if f.read(4) != MAGIC_NUMBER:
            raise ValueError("Неверная сигнатура")

        version = struct.unpack('B', f.read(1))[0]

        width = struct.unpack('I', f.read(4))[0]
        height = struct.unpack('I', f.read(4))[0]
        colorspace = struct.unpack('B', f.read(1))[0]
        quality = struct.unpack('B', f.read(1))[0]
        subsampling = bool(struct.unpack('B', f.read(1))[0])

        quant_y = np.array(
            [struct.unpack('f', f.read(4))[0] for _ in range(64)]
        ).reshape(8, 8)

        quant_c = np.array(
            [struct.unpack('f', f.read(4))[0] for _ in range(64)]
        ).reshape(8, 8)

        num_blocks = struct.unpack('I', f.read(4))[0]
        cb_blocks = struct.unpack('I', f.read(4))[0]
        cr_blocks = struct.unpack('I', f.read(4))[0]

        y_bitstream_len = struct.unpack('I', f.read(4))[0]
        cb_bitstream_len = struct.unpack('I', f.read(4))[0]

        bitstream_len = struct.unpack('I', f.read(4))[0]
        byte_data = f.read()

        bitstream = ''.join(format(b, '08b') for b in byte_data)
        bitstream = bitstream[:bitstream_len]

    return {
        'width': width,
        'height': height,
        'colorspace': colorspace,
        'quality': quality,
        'quant_table': quant_y,
        'quant_table_c': quant_c,
        'num_blocks': num_blocks,
        'cb_blocks': cb_blocks,
        'cr_blocks': cr_blocks,
        'y_bitstream_len': y_bitstream_len,
        'cb_bitstream_len': cb_bitstream_len,
        'bitstream': bitstream,
        'subsampling': subsampling
    }