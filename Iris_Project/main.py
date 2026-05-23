import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# LOAD IRIS DATASET
# =========================
df = sns.load_dataset("iris")

# =========================
# DISPLAY BASIC INFORMATION
# =========================

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns)

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# =========================
# SCATTER PLOT
# =========================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x='petal_length',
    y='petal_width',
    hue='species',
    data=df
)

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

plt.show()

# =========================
# HISTOGRAMS
# =========================

df.hist(figsize=(10,8))

plt.suptitle("Feature Distributions")

plt.show()

# =========================
# BOX PLOT
# =========================

plt.figure(figsize=(10,6))

sns.boxplot(data=df)

plt.title("Box Plot of Iris Features")

plt.show()