from neural_network import SimpleNeuralNetwork
import numpy as np

# Create the network
nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=1)

# Prepare your data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Train the network
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Make predictions
for input_data in X:
    prediction = nn.predict(input_data)
    print(f"Input: {input_data}, Prediction: {prediction}")