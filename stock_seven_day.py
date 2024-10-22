import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load your stock data
data = pd.read_csv('./data/aapl.csv', parse_dates=['date'])

# Filter for the symbol you want to predict (e.g., AAPL)
def prepare_data(df, symbol):
    stock_data = df[df['symbol'] == symbol].copy()
    stock_data = stock_data.sort_values('date')
    
    # Create features such as moving averages, volatility, etc.
    stock_data['ma7'] = stock_data['close'].rolling(window=7).mean()
    stock_data['ma21'] = stock_data['close'].rolling(window=21).mean()
    
    stock_data['price_diff'] = stock_data['close'] - stock_data['open']

    # Create a target that is 7 days in the future
    stock_data['close_7d_future'] = stock_data['close'].shift(-7)
    
    # Drop rows with NaN values (caused by rolling and shifting)
    stock_data.dropna(inplace=True)
    
    return stock_data

# Prepare the data for a specific stock symbol
symbol_data = prepare_data(data, 'AAPL')

# Define the features and target (predicting 'close_7d_future' using current day's data)
X = symbol_data[['open', 'price_diff', 'volume', 'ma21']]
y = symbol_data['close_7d_future']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale the features (important for LSTM and many other models)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale the target (y_train and y_test)
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Reshape the input for LSTM (samples, time steps, features)
# LSTM expects 3D input (batch size, time steps, number of features), 
# so we use 1 time step here because we're using just current day data for predictions.
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=50))
model.add(Dense(units=1))  # Output one predicted value (the close price 7 days in the future)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test_scaled))

# Make predictions
predictions = model.predict(X_test_scaled)

# Rescale the predictions back to the original scale
predictions_rescaled = scaler_y.inverse_transform(predictions)

# Rescale y_test back to the original scale
y_test_rescaled = scaler_y.inverse_transform(y_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
print(f"Mean Squared Error: {mse}")

# Plot actual vs predicted prices
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Actual Stock Price')
plt.plot(predictions_rescaled, label='Predicted Stock Price (7 days ahead)')
plt.title('Stock Price Prediction (7 days ahead)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
