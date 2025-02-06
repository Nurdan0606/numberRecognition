import numpy as np
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def as_array(x):
    arr = np.zeros(10)
    arr[x] = 1
    return arr

np.random.seed(990)

input_to_hidden_weights = 2*np.random.random((20, 784)) - 1
hidden_to_output_weights = 2*np.random.random((10, 20)) - 1

epochs = 100
for epoch in tqdm(range(epochs)):
    for mnist_file_name in tqdm(range(60000)):
        training_inputs = np.zeros((784, 1))
        training_outputs = np.zeros((1, 10))
                    
        with open(f"mnist_dataset/mnist_image_{mnist_file_name+1}.txt", 'r') as f:
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

        hidden_to_output_weights -= 0.001*(np.array(delta).reshape(10, 1) @ np.array(hidden.T).reshape(1, 20))

        delta_input = np.transpose(hidden_to_output_weights) @ err.T * (hidden*(1-hidden))
        input_to_hidden_weights -= 0.001 * delta_input @ np.transpose(input)

np.save('weight_ith.npy', input_to_hidden_weights)
np.save('weight_hto.npy', hidden_to_output_weights)
