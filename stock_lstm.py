import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load your data
data = pd.read_csv('./data/aapl.csv', parse_dates=['date'])

# Filter for the symbol you want to predict (e.g., AAPL)
def prepare_data(df, symbol):
    stock_data = df[df['symbol'] == symbol].copy()
    stock_data = stock_data.sort_values('date')
    stock_data['price_diff'] = stock_data['close'] - stock_data['open']
    stock_data['ma42'] = stock_data['close'].rolling(window=42).mean()
    stock_data['ma42'].fillna(method='bfill', inplace=True)  # Fill NaNs by backfilling
    stock_data['ma21'] = stock_data['close'].rolling(window=21).mean()
    stock_data['ma21'].fillna(method='bfill', inplace=True)  # Fill NaNs by backfilling
    #symbol_data['ma200'] = symbol_data['close'].rolling(window=200).mean()
    stock_data['returns'] = stock_data['close'].pct_change()
    return stock_data

symbol_data = prepare_data(data, 'AAPL')

# Select the relevant features (e.g., 'open', 'high', 'low', 'close', 'volume')
feature_columns = ['open', 'close', 'high', 'low', 'volume', 'price_diff', 'ma21', 'ma42']
symbol_data = symbol_data[feature_columns].values  # Convert to NumPy array

# Scale all features (open, high, low, close, volume)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(symbol_data)

# Function to create sequences for LSTM
def create_sequences_multivariate(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])  # Sequence of 'time_steps' rows with all features
        y.append(data[i, 3])  # The 'close' price is still the target (index 3 in feature_columns)
    return np.array(X), np.array(y)

# Create sequences with 60 time steps
time_steps = 60
X, y = create_sequences_multivariate(scaled_data, time_steps)

# Reshape data to be compatible with LSTM (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], len(feature_columns)))

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(Dropout(0.2))  # Dropout to prevent overfitting
model.add(LSTM(units=64, return_sequences=False))
#model.add(Dropout(0.2))
model.add(Dense(units=32))
model.add(Dense(units=1))  # Final output layer (predicting the 'close' price)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)

# Reshape predictions to match the input shape of inverse_transform
predictions_reshaped = np.zeros((len(predictions), len(feature_columns)))
predictions_reshaped[:, 3] = predictions.flatten()  # Put predictions in the 'close' column

# Inverse transform
predictions_rescaled = scaler.inverse_transform(predictions_reshaped)[:, 3]

# Rescale the true values back to the original scale
y_test_rescaled = scaler.inverse_transform(X_test[:, -1, :])[:, 3]

# Evaluate the model
mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
print(f"Mean Squared Error: {mse}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(predictions_rescaled, label='Predicted')
plt.title('LSTM Model - Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
