import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from compressor import compress_image, decompress_image


def analyze_quality_dependency(image_path, output_dir="output/analysis"):
    os.makedirs(output_dir, exist_ok=True)

    qualities = range(10, 101, 10)
    sizes = []

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\nАнализ изображения: {image_name}")
    print("=" * 60)

    for quality in qualities:
        compressed_path = f"{output_dir}/{image_name}_q{quality}.mjpg"
        decompressed_path = f"{output_dir}/{image_name}_q{quality}_dec.png"

        # Сжатие
        size = compress_image(image_path, compressed_path, quality=quality)
        sizes.append(size)

        # Декомпрессия
        decompress_image(compressed_path, decompressed_path)

        print(f"Quality {quality:3d}: {size:8d} байт")

    plt.figure(figsize=(10, 6))
    plt.plot(qualities, sizes, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Качество', fontsize=12)
    plt.ylabel('Размер файла (байт)', fontsize=12)
    plt.title(f'Зависимость размера от качества\n{image_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    graph_path = f"{output_dir}/{image_name}_quality_graph.png"
    plt.savefig(graph_path, dpi=150)
    plt.close()

    print(f"\nГрафик сохранен: {graph_path}")
    print("=" * 60)

    return qualities, sizes


def analyze_all_test_images(input_dir="input"):
    test_images = [
        "Lena.png",
        "color_image.png",
        "grayscale.png",
        "bw.png",
        "bw_dithered.png"
    ]

    results = {}

    for img_name in test_images:
        img_path = os.path.join(input_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Файл не найден: {img_path}")
            continue

        qualities, sizes = analyze_quality_dependency(img_path)
        results[img_name] = (qualities, sizes)

    # Сводный график
    plt.figure(figsize=(12, 7))

    for img_name, (qualities, sizes) in results.items():
        label = os.path.splitext(img_name)[0]
        plt.plot(qualities, sizes, marker='o', label=label, linewidth=2)

    plt.xlabel('Качество', fontsize=12)
    plt.ylabel('Размер файла (байт)', fontsize=12)
    plt.title('Сравнение размеров сжатых изображений', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    summary_path = "output/analysis/summary_comparison.png"
    plt.savefig(summary_path, dpi=150)
    plt.close()

    print(f"\nСводный график сохранен: {summary_path}")



if __name__ == "__main__":
    os.makedirs("output/analysis", exist_ok=True)
    analyze_all_test_images()