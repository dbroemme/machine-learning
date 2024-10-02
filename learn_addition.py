import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Generate dataset for addition
def generate_addition_data(num_samples=10000):
    X = np.random.randint(0, 100, size=(num_samples, 2))  # Two numbers as input
    y = np.sum(X, axis=1)  # Sum of the two numbers
    return X, y

# Deep learning approach
def build_deep_learning_model():
    model = Sequential([
        Dense(64, input_dim=2, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')  # Single output: the sum
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Generate data
X, y = generate_addition_data()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build and train the model
model = build_deep_learning_model()
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Save the trained model for future use
model.save('models/addition_model.h5')

# Test the model
test_input = np.array([[7, 3]])
deep_learning_sum = model.predict(test_input)
print(f"Deep Learning Addition: 7 + 3 = {deep_learning_sum[0][0]:.2f}")
