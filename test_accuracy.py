import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load realtime data
data = pd.read_csv("realtime_data.csv")

# Take first 20 rows
test_data = data.head(20)

# Remove timestamp column
test_data = test_data.iloc[:, 1:]

# Rename realtime columns
test_data = test_data.rename(columns={
    "BodyTemp": "temperature",
    "EnvTemp": "Ambient_Temperature",
    "HeartRate": "heart_rate"
})

# Add missing columns for scaler
test_data["SP_O2"] = 98
test_data["age"] = 25

# Exact scaler feature order
scaler_features = [
    "age",
    "heart_rate",
    "temperature",
    "SP_O2",
    "Ambient_Temperature"
]

test_data = test_data[scaler_features]

# Load scaler
scaler = joblib.load("scaler.pkl")

# Scale data
scaled_data = scaler.transform(test_data)

scaled_data = pd.DataFrame(
    scaled_data,
    columns=scaler_features
)

# Apply EWMA
ewma_data = scaled_data.ewm(span=10).mean()

# Remove age (model does not use it)
ewma_data = ewma_data.drop(columns=["age"])

# Rename columns to model names
ewma_data = ewma_data.rename(columns={
    "heart_rate": "heart rate",
    "SP_O2": "SP O2",
    "Ambient_Temperature": "Ambient Temperature (°C)"
})

# Exact model feature order (VERY IMPORTANT)
model_features = [
    "heart rate",
    "SP O2",
    "Ambient Temperature (°C)",
    "temperature"
]

ewma_data = ewma_data[model_features]

# Load model
model = joblib.load("isolation_forest_model.pkl")

# Predict
predictions = model.predict(ewma_data)

print("\nPredictions:")
print(predictions)

# Actual labels (change if needed)
actual_labels = [1] * 20

# Accuracy
accuracy = accuracy_score(actual_labels, predictions)

print("\nAccuracy:", accuracy * 100, "%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(actual_labels, predictions))

# Classification Report
print("\nClassification Report:")
print(classification_report(actual_labels, predictions))