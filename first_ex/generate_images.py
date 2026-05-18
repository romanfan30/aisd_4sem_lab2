# Задание 1.1: Подготовка набора тестовых данных

from PIL import Image
import numpy as np

# Укажи путь к своему цветному изображению
input_path = "../input/color_image.png"

# Загружаем изображение
img = Image.open(input_path).convert("RGB")

# 1. Оттенки серого
gray = img.convert("L")
gray.save("grayscale.png")

# 2. Черно-белое без дизеринга (round(c/255)*255)
gray_array = np.array(gray)
bw_array = (gray_array > 127) * 255
bw = Image.fromarray(bw_array.astype("uint8"))
bw.save("bw.png")

# 3. Черно-белое с дизерингом
bw_dithered = gray.convert("1")
bw_dithered.save("bw_dithered.png")
