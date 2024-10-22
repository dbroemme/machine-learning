import numpy as np
from neuron import Neuron

class MLP:
    def __init__(self, layer_sizes, activation_function):
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function
        self.layers = []
        
        for i in range(1, len(layer_sizes)):
            layer = [Neuron(layer_sizes[i-1], activation_function) for _ in range(layer_sizes[i])]
            self.layers.append(layer)
    
    def forward(self, inputs):
        for layer in self.layers:
            outputs = []
            for neuron in layer:
                # Pad or truncate inputs to match the number of weights
                adjusted_inputs = inputs[:neuron.num_inputs] + [0] * (neuron.num_inputs - len(inputs))
                outputs.append(neuron.forward(adjusted_inputs))
            inputs = outputs
        return inputs[0] if len(inputs) == 1 else inputs
    
    def set_weights_and_biases(self, weights, biases):
        for layer, layer_weights, layer_biases in zip(self.layers, weights, biases):
            for neuron, neuron_weights, neuron_bias in zip(layer, layer_weights, layer_biases):
                neuron.set_weights(neuron_weights)
                neuron.set_bias(neuron_bias)

def create_mlp_with_random_params(layer_sizes, activation_function):
    mlp = MLP(layer_sizes, activation_function)
    weights = [[np.random.randn(layer_sizes[i-1]) for _ in range(layer_sizes[i])] for i in range(1, len(layer_sizes))]
    biases = [[np.random.randn() for _ in range(layer_sizes[i])] for i in range(1, len(layer_sizes))]
    mlp.set_weights_and_biases(weights, biases)
    return mlp
