from PIL import Image
import numpy as np
from tqdm import tqdm
import struct


label_file_path = 'train-labels-idx1-ubyte'

def as_array(x):
    arr = np.zeros(10)
    arr[x] = 1
    return arr

def read_mnist_labels(label_file_path):
    with open(label_file_path, 'rb') as f:
        magic_number, num_labels = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

labels = read_mnist_labels(label_file_path)



def convert_image_to_text(input_file, output_file, ind):
    image = Image.open(input_file).convert("L")

    matrix = np.zeros((28, 28))

    for i in range(28):
        for j in range(28):
            pixel = 1 - image.getpixel((i, j))/255
    
            matrix[i][j] = ((pixel*100)//1)/100
    matrix = matrix.T

    with open(output_file, "w") as f:
        text = ""
        for i in matrix:
            for j in i:
                text += f"{j} "
            text += "\n"
        f.write(text + "\n" + f"{ind}")
        f.close()


for i in tqdm(range(60000)):
    convert_image_to_text(f"mnist_dataset/mnist_image_{i+1}.png", f"mnist_dataset/mnist_image_{i+1}.txt", f"{labels[i]}")

