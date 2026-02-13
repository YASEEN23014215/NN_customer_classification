# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:

### STEP 3:


## PROGRAM

### Name: 
### Register Number:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
        

```
```code
np.random.seed(42)
n = 2000

data = pd.DataFrame({
    "Age": np.random.randint(18, 70, n),
    "Annual_Income": np.random.randint(20000, 150000, n),
    "Spending_Score": np.random.randint(1, 100, n),
    "Family_Size": np.random.randint(1, 6, n),
    "Work_Experience": np.random.randint(0, 40, n)
})
conditions = [
    (data["Annual_Income"] > 90000) & (data["Spending_Score"] > 70),
    (data["Annual_Income"] > 60000) & (data["Spending_Score"] > 40),
    (data["Annual_Income"] > 40000),
]

choices = ["A", "B", "C"]
data["Segmentation"] = np.select(conditions, choices, default="D")

data.to_csv("customer_segmentation.csv", index=False)

print("Dataset Created Successfully!")
print(data.head())

X = data.drop("Segmentation", axis=1)
y = data["Segmentation"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", accuracy)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=label_encoder.classes_)
disp.plot()
plt.show()

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("Test Accuracy:", round(accuracy, 4))
print("Confusion Matrix:")
print(cm)
print("Name: YASEEN F")
print("Register No: 212223221026")
print("\nSample Predictions:")
for i in range(5):
    print("Predicted:",
          label_encoder.inverse_transform([y_pred[i]])[0],
          "| Actual:",
          label_encoder.inverse_transform([y_test[i]])[0])
```




## Dataset Information

<img width="710" height="303" alt="image" src="https://github.com/user-attachments/assets/97eeb60a-eca5-456f-a033-fe83428a1e1c" />

## OUTPUT
<img width="682" height="532" alt="image" src="https://github.com/user-attachments/assets/ad4643f3-0d8a-4438-b59f-0f28768a351f" />




### Confusion Matrix

<img width="269" height="153" alt="image" src="https://github.com/user-attachments/assets/287f9039-37b3-4fa2-b9b0-e943637696cd" />

### Classification Report

<img width="552" height="283" alt="image" src="https://github.com/user-attachments/assets/5c79152e-27df-4f63-bdbc-6b1edd92cec3" />


### New Sample Data Prediction

<img width="320" height="218" alt="image" src="https://github.com/user-attachments/assets/36d18aca-8dba-41c3-b4d9-8860398dbd05" />

## RESULT
The model was evaluated using Test Accuracy, Confusion Matrix, and Classification Report, and it was able to predict the customer segmentation (A, B, C, D) effectively.
