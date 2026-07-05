import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("finaldataset1.csv")

# Use features
X = data[['HeartRate', 'BodyTemp', 'AmbientTemp']]

# Clean normal data
X = X[
    (X["HeartRate"] > 40) &
    (X["HeartRate"] < 150) &
    (X["BodyTemp"] > 75) &
    (X["BodyTemp"] < 105) &
    (X["AmbientTemp"] > 5) &
    (X["AmbientTemp"] < 40)
]

# Split normal data
X_train, X_test_normal = train_test_split(
    X,
    test_size=0.2,
    random_state=42
)

# Create artificial anomalies (10% of test size)
n_anomaly = int(len(X_test_normal) * 0.1)

anomaly_data = pd.DataFrame({
    "HeartRate": np.random.uniform(160, 220, n_anomaly),
    "BodyTemp": np.random.uniform(105, 115, n_anomaly),
    "AmbientTemp": np.random.uniform(45, 60, n_anomaly)
})

# Combine normal + anomaly test data
X_test = pd.concat([X_test_normal, anomaly_data], ignore_index=True)

# True labels
# Normal = 1
# Anomaly = -1
y_true = [1] * len(X_test_normal) + [-1] * len(anomaly_data)

# Scale
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# EWMA smoothing
X_train_ewma = X_train_scaled.ewm(span=10).mean()
X_test_ewma = X_test_scaled.ewm(span=10).mean()

# Train Isolation Forest
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X_train_ewma)

# Predict
y_pred = model.predict(X_test_ewma)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

print("\nAccuracy:", accuracy * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))