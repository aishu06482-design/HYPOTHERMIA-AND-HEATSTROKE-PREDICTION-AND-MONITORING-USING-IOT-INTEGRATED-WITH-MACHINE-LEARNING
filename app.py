import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="IoT Patient Health Monitoring",
    layout="wide"
)

st.title("IoT Patient Health Monitoring Dashboard")

# AUTO REFRESH EVERY 2 SECONDS
st_autorefresh(interval=2000, limit=None, key="datarefresh")

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("isolation_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================================
# LOAD TRAINING DATASET
# ==========================================

dataset = pd.read_csv("cleaned_dataset_with_anomalies.csv")

# ==========================================
# SIDEBAR INPUT
# ==========================================

st.sidebar.header("Patient Information")

age = st.sidebar.number_input(
    "Enter Patient Age",
    min_value=1,
    max_value=100,
    value=25
)

# ==========================================
# LOAD REALTIME ESP32 DATA
# ==========================================

live_data = pd.read_csv("realtime_data.csv")

latest = live_data.iloc[-1]

heart = latest["HeartRate"]
temp = latest["BodyTemp"]
env = latest["EnvTemp"]
hum = latest["Humidity"]

# ==========================================
# MODEL INPUT
# ==========================================

spo2 = 98

input_data = pd.DataFrame(
    [[age, heart, temp, spo2, env]],
    columns=[
        "age",
        "heart_rate",
        "temperature",
        "SP_O2",
        "Ambient_Temperature"
    ]
)

scaled_input = scaler.transform(input_data)

prediction = model.predict(scaled_input)

# ==========================================
# DETERMINE CONDITION
# ==========================================

if heart == 0:
    status = "Waiting for Heart Rate Sensor..."

elif temp < 35:
    status = "Hypothermia"

elif temp > 39:
    status = "Heatstroke"

elif (55 <= heart <= 110) and (35 <= temp <= 38) and (15 <= env <= 35) and (20 <= hum <= 80):
    status = "Normal"

else:
    status = "Abnormal Condition"

# ==========================================
# LIVE PATIENT DATA
# ==========================================

st.header("Live Patient Monitoring")

c0, c1, c2, c3, c4, c5 = st.columns(6)

c0.metric("Age", age)
c1.metric("Heart Rate", f"{heart} BPM")
c2.metric("Body Temperature", f"{temp} °C")
c3.metric("Environment Temp", f"{env} °C")
c4.metric("Humidity", f"{hum} %")
c5.metric("Patient Status", status)

# ==========================================
# REAL TIME ANOMALY DETECTION
# ==========================================

st.subheader("Real-Time Anomaly Detection")

if prediction[0] == -1:
    st.error("⚠️ Anomaly Detected in Patient Health")
else:
    st.success("Patient Data Appears Normal")

st.divider()

# ==========================================
# DATASET TABLE
# ==========================================

st.subheader("1178 Patient Dataset")

st.dataframe(dataset)

# ==========================================
# CORRELATION MATRIX
# ==========================================

st.subheader("Feature Correlation Matrix")

corr = dataset[
    ["age", "heart_rate", "temperature", "SP_O2", "Ambient_Temperature"]
].corr()

fig_corr, ax = plt.subplots(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)

st.pyplot(fig_corr)

# ==========================================
# Z SCORE ANOMALY DETECTION
# ==========================================

st.subheader("Z Score Anomaly Detection")

dataset["z_score_temp"] = (
    dataset["temperature"] - dataset["temperature"].mean()
) / dataset["temperature"].std()

dataset["Z_Anomaly"] = dataset["z_score_temp"].apply(
    lambda x: 1 if abs(x) > 3 else 0
)

fig_z = px.scatter(
    dataset,
    x="temperature",
    y="heart_rate",
    color="Z_Anomaly",
    title="Z Score Based Anomaly Detection"
)

st.plotly_chart(fig_z, use_container_width=True)

# ==========================================
# ISOLATION FOREST VISUALIZATION
# ==========================================

st.subheader("Isolation Forest Anomaly Visualization")

fig_iso = px.scatter(
    dataset,
    x="temperature",
    y="heart_rate",
    color="ISO_Anomaly",
    title="Isolation Forest Detected Anomalies"
)

st.plotly_chart(fig_iso, use_container_width=True)

# ==========================================
# HEART RATE VS TEMPERATURE
# ==========================================

st.subheader("Heart Rate vs Temperature")

fig1 = px.scatter(
    dataset,
    x="temperature",
    y="heart_rate",
    color="ISO_Anomaly",
    title="Heart Rate vs Temperature"
)

st.plotly_chart(fig1, use_container_width=True)

# ==========================================
# HEART RATE DISTRIBUTION
# ==========================================

st.subheader("Heart Rate Distribution")

fig2 = px.histogram(
    dataset,
    x="heart_rate",
    nbins=40
)

st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# TEMPERATURE DISTRIBUTION
# ==========================================

st.subheader("Temperature Distribution")

fig3 = px.histogram(
    dataset,
    x="temperature",
    nbins=40
)

st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# PATIENT RISK DISTRIBUTION
# ==========================================

st.subheader("Patient Risk Distribution")

risk_counts = dataset["ISO_Anomaly"].value_counts()

risk_df = pd.DataFrame({
    "Condition": ["Normal", "Anomaly"],
    "Count": [
        risk_counts.get(0, 0),
        risk_counts.get(1, 0)
    ]
})

fig4 = px.pie(
    risk_df,
    names="Condition",
    values="Count"
)

st.plotly_chart(fig4)

# ==========================================
# LIVE SENSOR GRAPH
# ==========================================

st.subheader("Live Sensor Trend")

fig5 = px.line(
    live_data,
    x="Time",
    y=["HeartRate", "BodyTemp"]
)


st.plotly_chart(fig5, use_container_width=True)


st.set_page_config(page_title="Health Monitor", page_icon="🏥")

# ==========================================
# 🔧 DUMMY SENSOR VALUES (replace with your real ones)
# ==========================================
heart  = 88       # BPM
temp   = 36.8     # Body temperature °C
env    = 30.0     # Environment temperature °C
hum    = 65       # Humidity %
status = "Stable" # Patient status

# ==========================================
# 🤖 AI HEALTH ASSISTANT
# ==========================================
st.divider()
st.header("🤖 Caretaker AI Assistant")
st.info("Ask how to monitor or handle the patient condition.")

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# IMPROVED AI FUNCTION
# =========================
def caretaker_ai(q, status, heart, temp, env, hum):
    q = q.lower().strip()

    # --- Determine patient condition automatically ---
    if temp < 35:
        condition = "hypothermia"
    elif temp > 39:
        condition = "heatstroke"
    elif heart > 100:
        condition = "high heart rate"
    elif heart < 50:
        condition = "low heart rate"
    elif hum > 80:
        condition = "high humidity risk"
    else:
        condition = "stable"

    # --- Keyword matching (more keywords covered) ---
    if any(k in q for k in ["hypothermia", "cold", "shiver", "freezing", "low temp"]) or temp < 35:
        return f"""⚠️ HYPOTHERMIA ALERT
• Wrap patient in warm blankets immediately
• Move to a warm room away from cold
• Give warm (not hot) fluids if conscious
• Do NOT rub limbs — it can cause shock
• Call emergency if temp drops below 32°C
📊 Current Temp: {temp}°C | Heart Rate: {heart} BPM"""

    elif any(k in q for k in ["heatstroke", "hot", "fever", "overheating", "high temp", "heat"]) or temp > 39:
        return f"""⚠️ HEATSTROKE / HIGH FEVER ALERT
• Move patient to a cool, shaded area
• Apply cool wet cloth on forehead & neck
• Give small sips of water or ORS
• Use a fan or AC if available
• Seek medical help if temp exceeds 40°C
📊 Current Temp: {temp}°C | Env Temp: {env}°C"""

    elif any(k in q for k in ["heart", "pulse", "bpm", "heartbeat", "palpitation"]):
        if heart > 100:
            advice = "⚠️ HIGH — Ask patient to rest, take slow deep breaths"
        elif heart < 50:
            advice = "⚠️ LOW — Monitor closely, keep patient calm, seek help if dizzy"
        else:
            advice = "✅ NORMAL — Continue regular monitoring"
        return f"""💓 HEART RATE REPORT
• Current BPM: {heart}
• Status: {advice}
• Measure again in 5 minutes
• Log readings every 15 mins"""

    elif any(k in q for k in ["humidity", "humid", "moisture", "sweating", "sweat"]):
        if hum > 80:
            advice = "⚠️ HIGH — Risk of heat exhaustion. Use fan/AC, keep patient hydrated."
        elif hum < 30:
            advice = "⚠️ LOW — Risk of dehydration. Give fluids and use a humidifier."
        else:
            advice = "✅ NORMAL range."
        return f"""💧 HUMIDITY REPORT
• Current Humidity: {hum}%
• {advice}"""

    elif any(k in q for k in ["monitor", "check", "status", "reading", "vitals", "report", "update"]):
        return f"""📊 PATIENT VITALS REPORT
┌─────────────────────────────┐
│ 💓 Heart Rate  : {heart} BPM
│ 🌡️ Body Temp   : {temp}°C
│ 🌤️ Env Temp    : {env}°C
│ 💧 Humidity    : {hum}%
│ 🟢 Status      : {status}
└─────────────────────────────┘
✅ Check every 5–10 mins
📝 Log all readings in the record sheet"""

    elif any(k in q for k in ["safe", "okay", "ok", "condition", "how is", "fine", "danger"]):
        if condition == "stable":
            return f"""✅ PATIENT IS {status.upper()}
• All vitals are within normal range
• Continue regular monitoring
• Ensure patient is hydrated and comfortable
• No immediate action required"""
        else:
            return f"""⚠️ ATTENTION NEEDED
• Condition detected: {condition.upper()}
• Body Temp: {temp}°C | Heart: {heart} BPM
• Take appropriate action immediately
• Consider contacting medical help"""

    elif any(k in q for k in ["medicine", "medication", "drug", "dose", "tablet", "pill"]):
        return """💊 MEDICATION REMINDER
• Give medication only as prescribed by the doctor
• Do NOT skip doses
• Note the time of each dose
• Watch for allergic reactions (rash, breathing difficulty)
• Store medicines away from heat and humidity"""

    elif any(k in q for k in ["food", "eat", "diet", "nutrition", "drink", "water", "fluid"]):
        return f"""🥗 DIET & HYDRATION
• Give small, frequent meals
• Ensure at least 6–8 glasses of water daily
• Avoid oily, spicy, or heavy food
• Warm soups and fruits are recommended
• Current Env Temp is {env}°C — {'increase fluids if hot' if env > 30 else 'normal intake is fine'}"""

    elif any(k in q for k in ["emergency", "help", "urgent", "critical", "ambulance", "911", "serious"]):
        return """🚨 EMERGENCY PROTOCOL
1. CALL 108 (Ambulance) or nearest hospital immediately
2. Keep patient lying down, head slightly elevated
3. Do NOT give food/water if unconscious
4. Monitor breathing — start CPR if needed
5. Stay calm and keep the patient calm
6. Send someone to guide the ambulance to your location"""

    elif any(k in q for k in ["sleep", "rest", "tired", "fatigue", "energy", "weak"]):
        return f"""😴 REST & RECOVERY
• Ensure 7–9 hours of undisturbed sleep
• Keep room temperature comfortable (~24–26°C)
• Current env temp: {env}°C — {'consider cooling the room' if env > 28 else 'room temp is fine'}
• Reduce screen time before bedtime
• Gentle activity (short walks) helps recovery"""

    elif any(k in q for k in ["hi", "hello", "hey", "good morning", "good evening"]):
        return f"""👋 Hello! I'm the Caretaker AI Assistant.

I can help you with:
• Patient vitals & monitoring
• Fever, hypothermia, heatstroke guidance
• Heart rate & humidity alerts
• Diet, medication & emergency help

Current Status: {status} | Temp: {temp}°C | Heart: {heart} BPM
Type your question to get started!"""

    else:
        # ---- GENERAL FALLBACK with condition-aware advice ----
        base = f"""🤖 GENERAL HEALTH ADVICE
• Patient Status: {status}
• Body Temp: {temp}°C {'⚠️ HIGH' if temp > 39 else '⚠️ LOW' if temp < 35 else '✅ Normal'}
• Heart Rate: {heart} BPM {'⚠️ HIGH' if heart > 100 else '⚠️ LOW' if heart < 50 else '✅ Normal'}
• Humidity: {hum}% | Env Temp: {env}°C

📋 General Care Tips:
• Monitor vitals every 10–15 minutes
• Keep patient hydrated and comfortable
• Ensure good ventilation in the room
• Document all readings and changes
• Contact doctor if condition worsens"""
        if condition != "stable":
            base += f"\n\n⚠️ Note: Detected {condition.upper()} — please review urgently!"
        return base

# =========================
# INPUT
# =========================
user_input = st.text_input("💬 Type your question here...", key="input_box",
                            placeholder="e.g. How is the patient? / Check heart rate / Emergency help")

# =========================
# BUTTON
# =========================
col1, col2 = st.columns([1, 5])
with col1:
    ask_clicked = st.button("🔍 Ask AI", use_container_width=True)
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if ask_clicked:
    if user_input.strip() != "":
        reply = caretaker_ai(user_input, status, heart, temp, env, hum)
        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("AI", reply))
        st.success(reply)
    else:
        st.warning("⚠️ Please enter a question before clicking Ask AI.")

# =========================
# SHOW CHAT HISTORY
# =========================
if st.session_state.messages:
    st.divider()
    st.subheader("💬 Chat History")
    for sender, msg in reversed(st.session_state.messages):
        if sender == "You":
            st.markdown(f"🧑 **You:** {msg}")
        else:
            with st.container(border=True):
                st.markdown(f"🤖 **AI:** {msg}")