import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load realtime data
data = pd.read_csv("realtime_data.csv")

# Remove invalid HeartRate values
data = data[data["HeartRate"] != 0]

# Remove unrealistic values (clean realtime data)
data = data[
    (data["HeartRate"] > 40) &
    (data["HeartRate"] < 150) &
    (data["BodyTemp"] > 30) &
    (data["BodyTemp"] < 40)
]

# Rename EnvTemp to match training
data = data.rename(columns={
    "EnvTemp": "AmbientTemp"
})

# Select only same features used in training
X_test = data[
    ["HeartRate", "BodyTemp", "AmbientTemp"]
]

# Load scaler
scaler = joblib.load("scaler.pkl")

# Scale test data
X_scaled = scaler.transform(X_test)

# Convert back to dataframe
X_scaled = pd.DataFrame(
    X_scaled,
    columns=X_test.columns
)

# Apply EWMA smoothing
X_ewma = X_scaled.ewm(span=10).mean()

# Load trained model
model = joblib.load("isolation_model.pkl")

# Predict
predictions = model.predict(X_ewma)

print("\nPredictions:")
print(predictions)

# Number of valid rows
n = len(predictions)

print("\nTotal valid rows:", n)

# Assuming all valid rows are normal
actual_labels = [1] * n

# Accuracy
accuracy = accuracy_score(actual_labels, predictions)

print("\nAccuracy:", accuracy * 100, "%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(actual_labels, predictions))

# Classification Report
print("\nClassification Report:")
print(classification_report(actual_labels, predictions))