import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('./data/aapl.csv', parse_dates=['date'])

# Filter for a specific ticker symbol (AAPL for example)
def prepare_data(df, symbol):
    stock_data = df[df['symbol'] == symbol].copy()
    
    # Sort by date
    stock_data = stock_data.sort_values('date')
    
    # Feature engineering (e.g., moving averages, previous prices)
    stock_data['price_diff'] = stock_data['close'].shift(1) - stock_data['open'].shift(1)
    stock_data['ma7'] = stock_data['close'].shift(1).rolling(window=7).mean()
    stock_data['ma21'] = stock_data['close'].shift(1).rolling(window=21).mean()
    stock_data['volatility'] = (stock_data['high'].shift(1) - stock_data['low'].shift(1)) / stock_data['open'].shift(1)
    stock_data['yesterday_volume'] = stock_data['volume'].shift(1)
    stock_data['returns'] = stock_data['close'].pct_change()
    
    # Fill NaN values created by rolling operations
    stock_data = stock_data.fillna(0)
    
    return stock_data

symbol = 'AAPL'
# Prepare data for a specific stock symbol (e.g., AAPL)
symbol_data = prepare_data(data, symbol)

# Define the target and features
#X = symbol_data[['open', 'high', 'low', 'volume', 'ma7', 'ma21', 'volatility', 'returns']]
X = symbol_data[['open', 'ma7', 'ma21', 'volatility', 'yesterday_volume']]
y = symbol_data['close']

# Train-test split (use the earlier data to train, later data to test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(symbol_data['date'].iloc[-len(y_test):], y_test, label='Actual')
plt.plot(symbol_data['date'].iloc[-len(y_pred):], y_pred, label='Predicted')
plt.title(f'{symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
