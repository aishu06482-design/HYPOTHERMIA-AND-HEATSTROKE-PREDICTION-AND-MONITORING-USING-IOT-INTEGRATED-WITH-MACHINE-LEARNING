import streamlit as st
import pandas as pd
import joblib
from twilio.rest import Client

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(page_title="Health Monitoring", layout="wide")
st.title("IoT Patient Health Monitoring Dashboard")

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("health_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================================
# SIDEBAR INPUT
# ==========================================

st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", 1, 100, 25)

st.sidebar.subheader("Enter Patient Data")

heart = st.sidebar.number_input("Heart Rate (BPM)", 40, 180, 80)
temp = st.sidebar.number_input("Body Temperature (°C)", 30.0, 45.0, 36.5)
env = st.sidebar.number_input("Environment Temp (°C)", 10.0, 50.0, 30.0)
hum = st.sidebar.number_input("Humidity (%)", 0, 100, 60)
spo2 = st.sidebar.number_input("SpO2 (%)", 80, 100, 98)

# ==========================================
# MODEL INPUT
# ==========================================

input_data = pd.DataFrame(
    [[age, heart, temp, spo2, env]],
    columns=["age","heart_rate","temperature","SP_O2","Ambient_Temperature"]
)

scaled_input = scaler.transform(input_data)
prediction = model.predict(scaled_input)

# ==========================================
# STATUS LOGIC
# ==========================================

if temp < 35:
    status = "Hypothermia"
elif temp > 39:
    status = "Heatstroke"
elif prediction[0] == -1:
    status = "Abnormal"
else:
    status = "Normal"

# ==========================================
# ALERT CONTROL (IMPORTANT)
# ==========================================

if "alert_sent" not in st.session_state:
    st.session_state.alert_sent = False

if status == "Normal":
    st.session_state.alert_sent = False

# ==========================================
# DISPLAY
# ==========================================

st.header("Live Patient Monitoring")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Heart Rate", heart)
c2.metric("Temperature", temp)
c3.metric("Environment Temp", env)
c4.metric("Humidity", hum)
c5.metric("Status", status)

# ==========================================
# AI ASSISTANT
# ==========================================

def ai_assistant(status):
    if status == "Heatstroke":
        return "⚠️ High risk of heatstroke. Move patient to a cool place and hydrate immediately."
    elif status == "Hypothermia":
        return "⚠️ Risk of hypothermia. Keep patient warm and monitor closely."
    elif status == "Abnormal":
        return "⚠️ Abnormal vitals detected. Immediate medical attention advised."
    else:
        return "✅ Patient is stable."

st.subheader("AI Assistant")
st.info(ai_assistant(status))

# ==========================================
# 📞 TWILIO CALL ALERT
# ==========================================

def make_call():
    account_sid = "id"
    auth_token = "token"

    client = Client(account_sid, auth_token)

    call = client.calls.create(
        to="+919962032946",        # your number
        from_="+16624902894",   # Twilio number
        url="http://twimlets.com/message?Message=Patient%20is%20in%20critical%20condition"
    )

    return call.sid

# ==========================================
# TRIGGER CALL
# ==========================================

if status in ["Heatstroke","Hypothermia","Abnormal"]:
    if not st.session_state.alert_sent:
        make_call()
        st.session_state.alert_sent = True
        st.error("📞 Emergency Call Triggered!")

# ==========================================
# DONE
# ==========================================



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

# ==========================================
# LOAD DATASET
# ==========================================

try:
    df = pd.read_csv("finaldataset1.csv")
except:
    df = pd.read_excel("finaldataset1.xlsx")

print("Dataset Loaded Successfully")
print("Total Records:", len(df))

# ==========================================
# CLEAN COLUMN NAMES
# ==========================================

df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("/", "_")

# FIX SPECIAL CHARACTERS
df.columns = df.columns.str.replace("(°C)", "", regex=False)
df.columns = df.columns.str.replace("(", "", regex=False)
df.columns = df.columns.str.replace(")", "", regex=False)

df = df.rename(columns={"Ambient_Temperature_": "Ambient_Temperature"})

# ==========================================
# DROP ID
# ==========================================

df = df.drop(columns=["ID"], errors="ignore")

# ==========================================
# HANDLE MISSING
# ==========================================

df = df.ffill()
df = df.fillna(df.mean())

# ==========================================
# FILTER VALUES
# ==========================================

df = df[
    (df["heart_rate"].between(40, 180)) &
    (df["temperature"].between(30, 45)) &
    (df["SP_O2"].between(80, 100)) &
    (df["Ambient_Temperature"].between(10, 50))
]

# ==========================================
# EWMA
# ==========================================

df_ewma = df.copy()
for col in df.columns:
    df_ewma[col] = df[col].ewm(span=10).mean()

print("EWMA Smoothing Applied")

# ==========================================
# SCALE
# ==========================================

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_ewma)

print("Data Standardized")

# ==========================================
# Z SCORE
# ==========================================

z_scores = np.abs((scaled_data - scaled_data.mean(axis=0)) /
                  scaled_data.std(axis=0))

df["Z_Anomaly"] = (z_scores > 3).any(axis=1).astype(int)

print("Z-Score Detection Completed")

# ==========================================
# ISOLATION FOREST
# ==========================================

model = IsolationForest(n_estimators=200, contamination=0.05)
model.fit(scaled_data)

iso_pred = model.predict(scaled_data)
df["ISO_Anomaly"] = np.where(iso_pred == -1, 1, 0)

print("Isolation Forest Model Trained")

# ==========================================
# SAVE
# ==========================================

joblib.dump(model, "health_model.pkl")
joblib.dump(scaler, "scaler.pkl")

df.to_csv("processed_dataset.csv", index=False)

print("Model and Scaler Saved Successfully")
print("Processed Dataset Saved")
