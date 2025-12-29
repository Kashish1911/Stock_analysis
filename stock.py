# Install required libraries (run only once)
# pip install pandas numpy scikit-learn matplotlib yfinance

# Import required libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

ticker = "TSLA"

print("Downloading data...")
data = yf.download(ticker, start="2023-01-01", end="2026-12-01",auto_adjust=False)

# Check if data downloaded
if data.empty:
    print(" Error: Failed to download stock data. Check internet/DNS.")
    exit()


data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Forward fill missing values
data = data.ffill()

# Create label → next day's closing price
data['Next_Close'] = data['Close'].shift(-1)
data = data.dropna()

# Features & label
X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = data['Next_Close']

if len(data) < 10:
    print(" Not enough data to train the model.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = MLPRegressor(hidden_layer_sizes=(100, 100),
                     activation='relu',
                     solver='adam',
                     max_iter=500,
                     random_state=42)

print("Training model...")
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")


plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label="Actual Prices")
plt.plot(y_test.index, y_pred, label="Predicted Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Actual vs Predicted Closing Prices")
plt.legend()
plt.show()


def predict_next_close(model, scaler, new_data):
    new_data_scaled = scaler.transform(new_data)
    return model.predict(new_data_scaled)

# Example prediction
open_price = 170.00
high_price = 172.50
low_price = 169.00
close_price = 171.00
volume = 1000000

new_data = np.array([[open_price, high_price, low_price, close_price, volume]])
predicted_close = predict_next_close(model, scaler, new_data)

print(f"\nPredicted Next-Day Close Price: {predicted_close[0]}")
