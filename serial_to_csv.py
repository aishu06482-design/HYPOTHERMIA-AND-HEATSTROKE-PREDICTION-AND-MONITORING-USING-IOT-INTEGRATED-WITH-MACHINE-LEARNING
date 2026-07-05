import serial
import pandas as pd
from datetime import datetime
import os

ser = serial.Serial('COM3',115200,timeout=1)

filename = "realtime_data.csv"

print("Collecting ESP32 Data...")

# Create CSV with header if not exists
if not os.path.exists(filename):
    df = pd.DataFrame(columns=["Time","HeartRate","BodyTemp","EnvTemp","Humidity"])
    df.to_csv(filename,index=False)

while True:

    try:

        line = ser.readline().decode('utf-8',errors='ignore').strip()

        if "Heart Rate" not in line:
            continue

        print(line)

        parts = line.split("|")

        heart = float(parts[0].split(":")[1].replace("BPM","").strip())
        body = float(parts[1].split(":")[1].replace("C","").strip())
        env = float(parts[2].split(":")[1].replace("C","").strip())
        hum = float(parts[3].split(":")[1].replace("%","").strip())

        data = {
            "Time": datetime.now(),
            "HeartRate": heart,
            "BodyTemp": body,
            "EnvTemp": env,
            "Humidity": hum
        }

        df = pd.DataFrame([data])

        df.to_csv(filename,mode='a',header=False,index=False)


    except Exception as e:
        print("Skipping:",e)