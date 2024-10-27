import numpy as np

class Neuron:
    def __init__(self, input_size, activation_function):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.activation_function = activation_function

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.weights, inputs) + self.bias
        return self.activate(self.z)

    def activate(self, x):
        if self.activation_function == "ReLU":
            return max(0, x)
        elif self.activation_function == "Sigmoid":
            return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        if self.activation_function == "ReLU":
            return 1 if x > 0 else 0
        elif self.activation_function == "Sigmoid":
            return x * (1 - x)
