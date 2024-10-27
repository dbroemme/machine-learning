import numpy as np
from neuron import Neuron

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = [Neuron(input_size, "ReLU") for _ in range(hidden_size)]
        self.output_neuron = Neuron(hidden_size, "Sigmoid")

    def forward(self, input_data):
        self.hidden_outputs = [neuron.forward(input_data) for neuron in self.hidden_layer]
        return self.output_neuron.forward(self.hidden_outputs)

    def predict(self, input_data):
        return self.forward(input_data)

    def backward(self, input_data, target, learning_rate):
        # Forward pass
        output = self.forward(input_data)
        
        # Calculate output error
        output_error = target - output
        output_delta = output_error * self.output_neuron.activation_derivative(output)
        
        # Update output neuron weights
        for i in range(len(self.output_neuron.weights)):
            self.output_neuron.weights[i] += learning_rate * output_delta * self.hidden_outputs[i]
        self.output_neuron.bias += learning_rate * output_delta
        
        # Calculate hidden layer errors and update weights
        for i, hidden_neuron in enumerate(self.hidden_layer):
            hidden_error = output_delta * self.output_neuron.weights[i]
            hidden_delta = hidden_error * hidden_neuron.activation_derivative(self.hidden_outputs[i])
            
            for j in range(len(hidden_neuron.weights)):
                hidden_neuron.weights[j] += learning_rate * hidden_delta * input_data[j]
            hidden_neuron.bias += learning_rate * hidden_delta

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for input_data, target in zip(X, y):
                output = self.forward(input_data)
                loss = 0.5 * (target - output) ** 2
                total_loss += loss
                self.backward(input_data, target, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")
