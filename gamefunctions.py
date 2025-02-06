import numpy as np
from PIL import Image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def as_array(x):
    arr = np.zeros(10)
    arr[x-1] = 1
    return arr

def convert_image_to_text(input_file, output_file):
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
        f.write(text)
        f.close()
        
def add_txt_to_dataset(input_file : str, output_file : str, answer : str):
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
        text += "\n" + answer
        f.write(text)
        f.close()

def res():
    input_to_hidden_weights = np.load('weight_ith5.npy')
    hidden_to_output_weights = np.load('weight_hto5.npy')

    testi = np.zeros((784, 1))

    convert_image_to_text('drawing.png', 'drawing.txt')

    with open(f"drawing.txt", 'r') as f:
        text = f.read().split("\n")
        for i in range(28):
            data = list(map(float, text[i].split()))
            for j in range(28):
                testi[j+28*i] = data[j]

    return test(testi, input_to_hidden_weights, hidden_to_output_weights)

def train(input_file : str):
    input_to_hidden_weights = np.load('weight_ith.npy')
    hidden_to_output_weights = np.load('weight_hto.npy')

    training_inputs = np.zeros((784, 1))
    training_outputs = np.zeros((1, 10))
                
    with open(input_file, 'r') as f:
        text = f.read().split("\n")
        for i in range(28):
            data = list(map(float, text[i].split()))
            for j in range(28):
                training_inputs[j+28*i] = data[j]
        training_outputs = as_array(int(text[-1].split()[-1]))
                
    training_outputs = training_outputs.T

    input = training_inputs
    hidden = sigmoid(input_to_hidden_weights @ input)
    output = sigmoid(hidden_to_output_weights @ hidden)

    err = output.T - training_outputs.T

    delta = err*output.T*(1-output.T)

    hidden_to_output_weights -= 0.01*(np.array(delta).reshape(10, 1) @ np.array(hidden.T).reshape(1, 20))

    delta_input = np.transpose(hidden_to_output_weights) @ err.T * (hidden*(1-hidden))
    input_to_hidden_weights -= 0.01 * delta_input @ np.transpose(input)

    np.save('weight_ith.npy', input_to_hidden_weights)
    np.save('weight_hto.npy', hidden_to_output_weights)

def test(input, ithw, htow):
    hidden = sigmoid(np.dot(ithw, input))
    output = sigmoid(np.dot(htow, hidden))
    return output.T