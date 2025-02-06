import os
import struct
import numpy as np
from PIL import Image


file_path = 'train-images-idx3-ubyte'

def read_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic_number, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return image_data

def invert_image(image):
    inverted_image = 255 - image
    binary_image = np.where(inverted_image > 128, 255, 0).astype(np.uint8)
    
    return binary_image

def save_images(images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, image in enumerate(images):
        inverted_image = invert_image(image)
        img = Image.fromarray(inverted_image, mode='L')
        img.save(os.path.join(output_folder, f"mnist_image_{i+1}.png"))

images = read_mnist_images(file_path)

output_folder = 'mnist_dataset'

save_images(images, output_folder)

print(f"Все изображения сохранены в папку: {output_folder}")