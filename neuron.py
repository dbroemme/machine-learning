import numpy as np

class Neuron:
    def __init__(self, num_inputs, activation_function):
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs)
        self.bias = 0
        self.activation_function = activation_function

    def set_weights(self, weights):
        if len(weights) != self.num_inputs:
            raise ValueError("Number of weights must match number of inputs")
        self.weights = np.array(weights)

    def set_bias(self, bias):
        self.bias = bias

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        return np.maximum(0, x)

    def _tanh(self, x):
        return np.tanh(x)

    def get_weighted_sum(self, inputs):
        return np.dot(self.weights, inputs) + self.bias
    
    def forward(self, inputs):
        if len(inputs) != self.num_inputs:
            raise ValueError("Number of inputs must match number of weights")
        
        weighted_sum = self.get_weighted_sum(inputs)
        
        if self.activation_function == "Sigmoid":
            return self._sigmoid(weighted_sum)
        elif self.activation_function == "ReLU":
            return self._relu(weighted_sum)
        elif self.activation_function == "Tanh":
            return self._tanh(weighted_sum)
        else:
            raise ValueError("Unknown activation function")
