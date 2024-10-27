import numpy as np

class Neuron:
    def __init__(self, input_size, activation_function):
        self.num_inputs = input_size
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.activation_function = activation_function

    def set_weights(self, weights):
        if len(weights) != self.num_inputs:
            raise ValueError("Number of weights must match number of inputs")
        self.weights = np.array(weights)

    def set_bias(self, bias):
        self.bias = bias

    def get_weighted_sum(self, inputs):
        return np.dot(self.weights, inputs) + self.bias
    
    def forward(self, inputs):
        self.inputs = inputs
        self.z = self.get_weighted_sum(inputs)
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
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        return np.maximum(0, x)

    def _tanh(self, x):
        return np.tanh(x)
