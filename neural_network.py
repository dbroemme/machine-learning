import numpy as np
from neuron import Neuron
from sklearn.preprocessing import StandardScaler

# Set the random seed for NumPy
np.random.seed(42)  # You can choose any integer value

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, use_standard_scalar=True):
        self.use_standard_scalar = use_standard_scalar
        self.hidden_layer = [Neuron(input_size, "ReLU") for _ in range(hidden_size)]
        # TODO we are ignoring the output size, and right now just creating one output
        self.output_neuron = Neuron(hidden_size, "Sigmoid")
        if self.use_standard_scalar:
            self.input_scaler = StandardScaler()
            self.output_scaler = StandardScaler()
        else:
            self.input_min = None
            self.input_max = None
            self.output_min = None
            self.output_max = None

    def scale_input(self, input_data):
        if self.use_standard_scalar:
            # Convert input_data to numpy array if it's a list
            input_data = np.array(input_data)
            # Reshape input_data to 2D if it's 1D
            if input_data.ndim == 1:
                input_data = input_data.reshape(-1, 1)
            return self.input_scaler.transform(input_data)
        else:
            return (input_data - self.input_min) / (self.input_max - self.input_min)

    def descale_output(self, output):
        if self.use_standard_scalar:
            # Reshape output to 2D if it's 0D or 1D
            output = np.array(output)
            if output.ndim == 0:
                output = output.reshape(1, -1)
            elif output.ndim == 1:
                output = output.reshape(-1, 1)
            return self.output_scaler.inverse_transform(output).flatten()
        else:
            return output * (self.output_max - self.output_min) + self.output_min

    def fit_scalers(self, X, y):
        if self.use_standard_scalar:
            # Convert X and y to numpy arrays if they're lists
            X = np.array(X)
            y = np.array(y)
            # Reshape X to 2D if it's 1D
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            # Reshape y to 2D if it's 1D
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            self.input_scaler.fit(X)
            self.output_scaler.fit(y)
        else:
            self.input_min = np.min(X, axis=0)
            self.input_max = np.max(X, axis=0)
            self.output_min = np.min(y)
            self.output_max = np.max(y)

    def forward(self, input_data):
        scaled_input = self.scale_input(input_data)
        if self.use_standard_scalar:
            self.hidden_outputs = [neuron.forward(scaled_input.flatten()) for neuron in self.hidden_layer]
        else:
            self.hidden_outputs = [neuron.forward(scaled_input) for neuron in self.hidden_layer]
        return self.output_neuron.forward(self.hidden_outputs)

    def predict(self, input_data):
        output = self.forward(input_data)
        return self.descale_output(np.array([output]))


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
        if self.use_standard_scalar:
            self.fit_scalers(X, y)
            scaled_X = self.input_scaler.transform(np.array(X).reshape(-1, 1))
            scaled_y = self.output_scaler.transform(np.array(y).reshape(-1, 1)).flatten()
        else:
            self.fit_scalers(X, y)
            scaled_X = X
            scaled_y = (y - self.output_min) / (self.output_max - self.output_min)

        for epoch in range(epochs):
            total_loss = 0
            for input_data, target in zip(scaled_X, scaled_y):
                output = self.forward(input_data)
                loss = 0.5 * (target - output) ** 2
                total_loss += loss
                self.backward(input_data, target, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")
