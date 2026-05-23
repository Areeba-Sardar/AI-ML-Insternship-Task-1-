# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("heart_disease_uci.csv")

# =========================
# DISPLAY BASIC INFO
# =========================

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== INFO =====")
print(df.info())

print("\n===== STATISTICS =====")
print(df.describe())

# =========================
# HANDLE MISSING VALUES
# =========================

df = df.dropna()

print("\nMissing values removed!")

# =========================
# ENCODE CATEGORICAL COLUMNS
# =========================

label_encoder = LabelEncoder()

# Text columns
text_columns = [
    'sex',
    'dataset',
    'cp',
    'restecg',
    'slope',
    'thal'
]

# Encode text columns
for column in text_columns:

    df[column] = label_encoder.fit_transform(
        df[column].astype(str)
    )

# Encode boolean/object columns
df['fbs'] = label_encoder.fit_transform(
    df['fbs'].astype(str)
)

df['exang'] = label_encoder.fit_transform(
    df['exang'].astype(str)
)

print("\n===== DATA TYPES AFTER ENCODING =====")
print(df.dtypes)

# =========================
# TARGET COLUMN
# =========================

# 0 = No Disease
# 1 = Disease

df['num'] = df['num'].apply(
    lambda x: 1 if x > 0 else 0
)

# =========================
# HEART DISEASE COUNT PLOT
# =========================

plt.figure(figsize=(6,4))

sns.countplot(
    x='num',
    data=df
)

plt.title("Heart Disease Count")

plt.xlabel("Heart Disease")
plt.ylabel("Count")

plt.show()

# =========================
# CORRELATION HEATMAP
# =========================

numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric_df.corr(),
    annot=False,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# =========================
# FEATURES AND TARGET
# =========================

X = df.drop("num", axis=1)

y = df["num"]

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

model = LogisticRegression(max_iter=1000)

# =========================
# TRAIN MODEL
# =========================

model.fit(X_train, y_train)

print("\n===== MODEL TRAINING COMPLETE =====")

# =========================
# PREDICTIONS
# =========================

predictions = model.predict(X_test)

# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n===== ACCURACY =====")
print(accuracy)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =========================
# ROC CURVE
# =========================

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

auc_score = roc_auc_score(
    y_test,
    y_prob
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.2f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_[0]
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")
print(importance)