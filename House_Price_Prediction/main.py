# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import numpy as np

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("Housing.csv")

# =========================
# DISPLAY DATA
# =========================

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== DATA INFO =====")
print(df.info())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# =========================
# HANDLE CATEGORICAL COLUMNS
# =========================

label_encoder = LabelEncoder()

categorical_columns = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea',
    'furnishingstatus'
]

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# =========================
# FEATURES AND TARGET
# =========================

X = df.drop("price", axis=1)

y = df["price"]

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
# PREDICTIONS
# =========================

predictions = model.predict(X_test)

# =========================
# EVALUATION
# =========================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

print("\n===== MAE =====")
print(mae)

print("\n===== RMSE =====")
print(rmse)

# =========================
# ACTUAL VS PREDICTED GRAPH
# =========================

plt.figure(figsize=(10,6))

plt.scatter(
    y_test,
    predictions
)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.show()