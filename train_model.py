import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

df = pd.read_csv("finaldataset1.csv")

print("\nTotal Rows:", df.shape[0])
print("Original Columns:", df.columns)

# ==========================================
# STEP 2: CLEAN COLUMN NAMES (IMPORTANT FIX)
# ==========================================

df.columns = df.columns.str.strip()           # remove extra spaces
df.columns = df.columns.str.replace(" ", "_")  # replace space with _
df.columns = df.columns.str.replace("/", "_")

print("Cleaned Columns:", df.columns)

# ==========================================
# STEP 3: DROP ID COLUMN
# ==========================================

df = df.drop(columns=["ID"], errors="ignore")

# ==========================================
# STEP 4: HANDLE MISSING VALUES
# ==========================================

print("\nMissing values before cleaning:")
print(df.isnull().sum())

df = df.ffill()
df = df.fillna(df.mean())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ==========================================
# STEP 5: REMOVE IMPROPER SENSOR VALUES
# ==========================================

initial_rows = len(df)

df = df[
    (df["heart_rate"].between(40, 180)) &
    (df["temperature"].between(30, 45)) &
    (df["SP_O2"].between(80, 100)) &
    (df["Ambient_Temperature"].between(10, 50))
]

print("\nRows before filtering:", initial_rows)
print("Rows after filtering:", len(df))

# ==========================================
# STEP 6: EWMA SMOOTHING
# ==========================================

df_ewma = df.copy()

for col in df.columns:
    df_ewma[col] = df[col].ewm(span=10, adjust=False).mean()

print("EWMA smoothing applied")

# ==========================================
# STEP 7: STANDARDIZATION
# ==========================================

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_ewma)

# ==========================================
# STEP 8: Z-SCORE
# ==========================================

z_scores = np.abs((scaled_data - scaled_data.mean(axis=0)) /
                  scaled_data.std(axis=0))

threshold = 3
z_anomaly = (z_scores > threshold).any(axis=1)

df["Z_Anomaly"] = z_anomaly.astype(int)

print("Total Z-Score Anomalies:", df["Z_Anomaly"].sum())

# ==========================================
# STEP 9: ISOLATION FOREST
# ==========================================

iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

iso.fit(scaled_data)
iso_pred = iso.predict(scaled_data)

df["ISO_Anomaly"] = np.where(iso_pred == -1, 1, 0)

print("Total Isolation Forest Anomalies:", df["ISO_Anomaly"].sum())

# ==========================================
# STEP 10: SAVE MODELS
# ==========================================

joblib.dump(iso, "isolation_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Models saved successfully")

# ==========================================
# STEP 11: SAVE FINAL DATA
# ==========================================

df.to_csv("cleaned_dataset_with_anomalies.csv", index=False)

print("Final dataset saved")

# ==========================================
# STEP 12: VISUALIZATION
# ==========================================

plt.figure()
plt.plot(df["heart_rate"])
plt.scatter(df.index[df["ISO_Anomaly"] == 1],
            df["heart_rate"][df["ISO_Anomaly"] == 1])
plt.title("Isolation Forest - Heart Rate Anomalies")
plt.show()

plt.figure()
plt.plot(df["heart_rate"])
plt.scatter(df.index[df["Z_Anomaly"] == 1],
            df["heart_rate"][df["Z_Anomaly"] == 1])
plt.title("Z-Score - Heart Rate Anomalies")
plt.show()

print("\n===== TRAINING COMPLETED SUCCESSFULLY =====")