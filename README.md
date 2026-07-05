# 🩺 Hypothermia and Heatstroke Prediction and Monitoring Using IoT Integrated with Machine Learning

An AI-powered IoT healthcare monitoring system that detects hypothermia and heatstroke using real-time sensor data, machine learning, anomaly detection, an AI Health Assistant, and an Emergency Calling API. The system provides continuous patient monitoring through an interactive Streamlit dashboard, enabling early detection of critical health conditions and timely emergency response.

---

## 📌 Project Overview

Hypothermia and heatstroke are potentially life-threatening conditions that require immediate detection and intervention. This team project integrates IoT sensors, Machine Learning, Artificial Intelligence, and real-time visualization to continuously monitor patient health.

The system collects physiological data from IoT sensors connected to an ESP32 microcontroller, processes the data using Machine Learning algorithms, detects abnormal conditions through anomaly detection, provides AI-generated healthcare guidance, and triggers emergency calls to caregivers during critical situations.

## 🚀 Project Highlights

- Built using IoT, Machine Learning, and Artificial Intelligence
- Real-time patient health monitoring with ESP32 and biomedical sensors
- AI-powered healthcare assistant for user interaction
- Automatic emergency voice call alerts using Twilio API
- Interactive Streamlit dashboard for data visualization
- Isolation Forest-based anomaly detection for identifying abnormal health conditions

> **Note:** This repository showcases my contributions to our academic team project.

---

# 👩‍💻 My Contributions

- 🔌 Designed and implemented the IoT hardware connections using the ESP32, MAX30102, DS18B20, and DHT22 sensors.
- 📊 Collected, organized, and prepared the healthcare dataset used for Machine Learning training and testing.
- 🧹 Performed data preprocessing and prepared datasets for model development.
- 📞 Developed and integrated the Emergency Calling API using Twilio for automatic caregiver alerts during emergency situations.
- 🧪 Assisted in testing, debugging, and validating the integrated system for reliable real-time monitoring.

---

# ✨ Features

- 🌡️ Real-time Body Temperature Monitoring
- ❤️ Heart Rate Monitoring
- 💧 SpO₂ Monitoring
- 🌤️ Ambient Temperature & Humidity Monitoring
- 📊 Interactive Streamlit Dashboard
- 🤖 AI Health Assistant
- 🧠 Machine Learning-based Health Prediction
- ⚠️ Isolation Forest Anomaly Detection
- 📞 Emergency Calling API
- 📈 Real-time Data Visualization
- 📂 Historical Data Analysis

---

# 🛠️ Technologies Used

### Programming & Development
- Python
- Streamlit
- Git
- GitHub
- VS Code

### Machine Learning & Data Processing
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Isolation Forest
- Data Preprocessing
- Feature Scaling

### Data Visualization
- Plotly
- Matplotlib

### IoT Hardware
- ESP32
- MAX30102 (Heart Rate & SpO₂ Sensor)
- DS18B20 (Body Temperature Sensor)
- DHT22 (Temperature & Humidity Sensor)

### AI Health Assistant
- OpenAI API
- GPT-based AI Assistant
- Prompt Engineering
- Streamlit Chat Interface

### Emergency Alert System
- Twilio API
- Voice Call Integration

### Communication
- PySerial

---

# 📂 Project Structure

```
HYPOTHERMIA PROJECT
│
├── accuracy/
├── dataset/
├── screenshots/
├── app.py
├── train_model.py
├── test_accuracy.py
├── serial_to_csv.py
├── callingapi.py
├── README.md
└── requirements.txt
```

---

# ⚙️ System Workflow

1. IoT sensors continuously collect patient health data.
2. ESP32 transmits sensor readings to the computer.
3. Sensor data is preprocessed.
4. Machine Learning predicts the patient's health condition.
5. Isolation Forest identifies abnormal patterns.
6. The Streamlit dashboard displays live patient information.
7. The AI Health Assistant provides healthcare guidance.
8. The Emergency Calling API notifies caregivers during critical situations.

---

# 🧠 Machine Learning

The machine learning pipeline includes:

- Data Cleaning
- Data Preprocessing
- Feature Scaling
- Model Training
- Model Evaluation
- Isolation Forest-based Anomaly Detection
- Health Condition Prediction

---

# 🤖 AI Health Assistant

The integrated AI Health Assistant can:

- Explain patient health conditions
- Answer healthcare-related questions
- Suggest precautions for hypothermia and heatstroke
- Provide first-aid recommendations
- Assist caregivers in monitoring patients

---

# 🚨 Emergency Calling API

The system automatically alerts caregivers during emergency situations using the Twilio API.

### Features

- Automatic emergency voice calls
- Caregiver notifications
- Rapid emergency response

---

# 🚀 Installation

## Clone the repository

```bash
git clone https://github.com/aishu06482-design/HYPOTHERMIA-AND-HEATSTROKE-PREDICTION-AND-MONITORING-USING-IOT-INTEGRATED-WITH-MACHINE-LEARNING.git
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

---

# 🎯 Future Enhancements

- Mobile Application
- Cloud Database Integration
- Multi-patient Monitoring
- Wearable Device Support
- Remote Patient Monitoring
- Predictive Healthcare Analytics
- Doctor Dashboard
- Cloud Deployment

---

# 📄 License

This project was developed as an academic team project for educational purposes.

---

⭐ If you found this project interesting, consider giving it a star on GitHub.