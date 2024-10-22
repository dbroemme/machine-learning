import numpy as np
import pandas as pd
from hmmlearn import hmm

# Load stock data (e.g., AAPL)
data = pd.read_csv('./data/aapl.csv', parse_dates=['date'])

# Feature selection (you can use more features)
features = data[['close', 'volume']].values

# Train HMM with 2 hidden states (for example, bullish and bearish regimes)
model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
model.fit(features)

# Predict hidden states (regimes)
hidden_states = model.predict(features)

# Add hidden states to the dataframe
data['hidden_state'] = hidden_states

# Visualize the hidden states along with stock prices
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(data['date'], data['close'], label='Stock Price')
plt.scatter(data['date'], data['close'], c=data['hidden_state'], cmap='viridis', label='Hidden State', alpha=0.6)
plt.title('Stock Price and Hidden Market Regimes')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
