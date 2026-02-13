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
Understand the Problem

### STEP 2:
We need to predict which segment (A, B, C, or D) a new customer belongs to.

### STEP 3:
Load the Dataset
Import required libraries.
Load the dataset from the CSV file (from GitHub or local file).

### STEP 4:
Identify input features and target column (Segmentation).
### STEP 5:
Train the model using training data and Test the model using test data.
## STEP 6:
Check confusion matrix to see performance for each segment.


## PROGRAM

### Name: YASEEN F
### Register Number:21223220126
```

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

np.random.seed(42)

n = 2000

data = pd.DataFrame({
    "Age": np.random.randint(18, 70, n),
    "Annual_Income": np.random.randint(20000, 150000, n),
    "Spending_Score": np.random.randint(1, 100, n),
    "Family_Size": np.random.randint(1, 6, n),
    "Work_Experience": np.random.randint(0, 40, n)
})

# Create segmentation based on some pattern
conditions = [
    (data["Annual_Income"] > 90000) & (data["Spending_Score"] > 70),
    (data["Annual_Income"] > 60000) & (data["Spending_Score"] > 40),
    (data["Annual_Income"] > 40000),
]

choices = ["A", "B", "C"]
data["Segmentation"] = np.select(conditions, choices, default="D")

# Save CSV
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



loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)

print("Test Accuracy:", round(accuracy, 4))
print("Confusion Matrix:")
print(cm)


print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))
print("Name: YASEEN F")
print("Register No: 212223220126")
print("\nSample Predictions:")
for i in range(5):
    print("Predicted:",
          label_encoder.inverse_transform([y_pred[i]])[0],
          "| Actual:",
          label_encoder.inverse_transform([y_test[i]])[0])
```




## Dataset Information:
<img width="699" height="312" alt="image" src="https://github.com/user-attachments/assets/81518f4a-f6e2-413a-a09f-e7e39e819e05" />


## OUTPUT:

<img width="742" height="603" alt="image" src="https://github.com/user-attachments/assets/73c528ea-bf67-4066-8b49-f786634c022a" />


### Confusion Matrix:
<img width="228" height="128" alt="image" src="https://github.com/user-attachments/assets/f77ee8ad-21dd-4362-8a9a-cc215dbc6159" />


### Classification Report:

<img width="574" height="283" alt="image" src="https://github.com/user-attachments/assets/3a29f08d-f0a6-4ad1-a264-5255dbdb24f2" />


### New Sample Data Prediction:
<img width="328" height="211" alt="image" src="https://github.com/user-attachments/assets/0d7352fd-ca97-40ea-9b73-ecd10d259a89" />


## RESULT:
The model was evaluated using Test Accuracy, Confusion Matrix, and Classification Report, and it was able to predict the customer segmentation (A, B, C, D) effectively.
