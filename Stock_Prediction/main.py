# =========================
# IMPORT LIBRARIES
# =========================

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# =========================
# DOWNLOAD STOCK DATA
# =========================

print("Downloading stock data...")

data = yf.download(
    "AAPL",
    start="2023-01-01",
    end="2024-01-01"
)

# =========================
# DISPLAY DATA
# =========================

print("\n===== FIRST 5 ROWS =====")
print(data.head())

print("\n===== SHAPE =====")
print(data.shape)

print("\n===== DATA INFO =====")
print(data.info())

# =========================
# SELECT FEATURES
# =========================

X = data[['Open', 'High', 'Low', 'Volume']]

# TARGET
y = data['Close']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# CREATE MODEL
# =========================

model = LinearRegression()

# TRAIN MODEL
model.fit(X_train, y_train)

print("\nModel Training Complete!")

# =========================
# MAKE PREDICTIONS
# =========================

predictions = model.predict(X_test)

# =========================
# EVALUATION
# =========================

mae = mean_absolute_error(y_test, predictions)

print("\n===== MEAN ABSOLUTE ERROR =====")
print(mae)

# =========================
# ACTUAL VS PREDICTED GRAPH
# =========================

plt.figure(figsize=(12,6))

plt.plot(
    y_test.values,
    label='Actual Prices'
)

plt.plot(
    predictions,
    label='Predicted Prices'
)

plt.title("Actual vs Predicted Stock Prices")

plt.xlabel("Days")

plt.ylabel("Closing Price")

plt.legend()

plt.show()