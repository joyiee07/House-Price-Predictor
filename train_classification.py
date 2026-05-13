import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Sample dataset
data = {
    "study_hours": [1,2,3,4,5,6,7,8,9,10],
    "attendance": [50,55,60,65,70,75,80,85,90,95],
    "assignments": [2,2,3,3,4,4,5,5,5,5],
    "result": ["Fail","Fail","Fail","Fail","Pass","Pass","Pass","Pass","Pass","Pass"]
}

df = pd.DataFrame(data)

# Convert target to numbers
df["result"] = df["result"].map({"Fail":0, "Pass":1})

# Features and target
X = df[["study_hours", "attendance", "assignments"]]
y = df["result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Save model
joblib.dump(model, "student_classification_model.pkl")