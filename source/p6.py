import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
X, y = spiral_data(100,3)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        print(self.weights)
        print(self.biases)
    def foward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)
layer1.foward(X)
# layer2 = Layer_Dense(5,2)
# layer2.foward(layer1.output)

activation1 = Activation_ReLU()
activation1.forward(layer1.output)
# activation1.forward(layer2.output)
print(activation1.output)
