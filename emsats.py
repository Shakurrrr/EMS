# EMSATS attendance system code

import RPi.GPIO as GPIO
import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
import time
from picamera2 import Picamera2
from RPLCD.i2c import CharLCD
import threading
import psycopg2
import requests
from io import BytesIO
from PIL import Image
from apscheduler.schedulers.background import BackgroundScheduler

# PostgreSQL connection
pg_conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="emsats_db",
    user="emsats_user",
    password="mypassword123"
)
pg_conn.autocommit = True
pg_cursor = pg_conn.cursor()

# Constants and Paths
BASE_DIR = "/home/pi/Desktop/PROJECT/N-facial-recognition-QRCODE"
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pickle")
LOG_FILE = os.path.join(BASE_DIR, "employee_attendance_log.xlsx")
WEEKLY_LOG_FILE = os.path.join(BASE_DIR, "weekly_log.xlsx")
MONTHLY_LOG_FILE = os.path.join(BASE_DIR, "monthly_log.xlsx")
UNSYNCED_FILE = os.path.join(BASE_DIR, "unsynced_records.xlsx")

# GPIO
CHECK_IN_PIN = 22
CHECK_OUT_PIN = 23
GREEN_LED_PIN = 27
RED_LED_PIN = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(CHECK_IN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(CHECK_OUT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.output(GREEN_LED_PIN, GPIO.LOW)
GPIO.output(RED_LED_PIN, GPIO.LOW)

# LCD
lcd = CharLCD('PCF8574', 0x27)

# Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Globals
cv_scaler = 2
recent_logs = {}
known_face_encodings = []
known_face_names = []

# Load logs
try:
    log_df = pd.read_excel(LOG_FILE)
except FileNotFoundError:
    log_df = pd.DataFrame(columns=["Name", "EmployeeID", "Department", "Type", "Date", "Time", "Timestamp"])

def fetch_and_update_employees():
    global known_face_encodings, known_face_names
    pg_cursor.execute("SELECT name, image_url FROM employees")
    rows = pg_cursor.fetchall()
    names, encodings = [], []
    for name, image_url in rows:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            img_np = np.array(image)
            face_locations = face_recognition.face_locations(img_np)
            face_encodings = face_recognition.face_encodings(img_np, face_locations)
            if face_encodings:
                names.append(name)
                encodings.append(face_encodings[0])
        except Exception as e:
            print(f"[ERROR] Image processing for {name}: {e}")
    known_face_encodings = encodings
    known_face_names = names
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)

def background_employee_sync():
    while True:
        fetch_and_update_employees()
        time.sleep(300)

def save_unsynced_record(record):
    try:
        df_unsynced = pd.read_excel(UNSYNCED_FILE)
    except FileNotFoundError:
        df_unsynced = pd.DataFrame(columns=record.keys())
    df_unsynced = pd.concat([df_unsynced, pd.DataFrame([record])], ignore_index=True)
    df_unsynced.to_excel(UNSYNCED_FILE, index=False)

def upload_unsynced_records():
    try:
        df_unsynced = pd.read_excel(UNSYNCED_FILE)
    except FileNotFoundError:
        return
    success = []
    for i, row in df_unsynced.iterrows():
        try:
            pg_cursor.execute("""
                INSERT INTO employee_attendance (name, employee_id, department, type, date, time, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (row["Name"], row["EmployeeID"], row["Department"], row["Type"], row["Date"], row["Time"], row["Timestamp"]))
            success.append(i)
        except Exception as e:
            print(f"[SYNC ERROR] {e}")
    if success:
        df_unsynced.drop(success, inplace=True)
        df_unsynced.to_excel(UNSYNCED_FILE, index=False)

def background_offline_sync():
    while True:
        upload_unsynced_records()
        time.sleep(300)

def recognize_face(timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        frame = picam2.capture_array()
        resized = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)
        for encoding in encodings:
            matches = face_recognition.compare_faces(known_face_encodings, encoding)
            distances = face_recognition.face_distance(known_face_encodings, encoding)
            if matches and distances.size:
                i = np.argmin(distances)
                if matches[i]:
                    return known_face_names[i], round((1 - distances[i]) * 100, 2)
    return None, None

def log_attendance(name, log_type):
    global log_df
    pg_cursor.execute("SELECT employee_id, department FROM employees WHERE name = %s LIMIT 1", (name,))
    row = pg_cursor.fetchone()
    if not row:
        return False
    employee_id, department = row
    now = datetime.now()
    timestamp = now.timestamp()
    record = {
        "Name": name,
        "EmployeeID": employee_id,
        "Department": department,
        "Type": log_type,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Timestamp": timestamp
    }
    if name in recent_logs and (now - recent_logs[name]).total_seconds() < 900:
        return "duplicate"
    log_df = pd.concat([log_df, pd.DataFrame([record])], ignore_index=True)
    log_df.to_excel(LOG_FILE, index=False)
    try:
        pg_cursor.execute("""
            INSERT INTO employee_attendance (name, employee_id, department, type, date, time, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (name, employee_id, department, log_type, record["Date"], record["Time"], record["Timestamp"]))
    except Exception as e:
        print(f"[DB ERROR] {e}, saving locally.")
        save_unsynced_record(record)
    recent_logs[name] = now
    return record

def display_message(lines):
    lcd.clear()
    for line in lines:
        lcd.write_string(line.ljust(16))
        lcd.crlf()

def process_attendance(log_type):
    display_message(["Look at the camera"])
    name, confidence = recognize_face()
    if name:
        display_message([f"Detected: {name}", f"Match: {confidence}%"])
        result = log_attendance(name, log_type)
        time.sleep(1)
        if result == "duplicate":
            display_message(["Already", f"Checked {log_type}".lower()])
        elif result:
            msg = [f"Welcome {name.split()[0]}" if log_type == "Check-In" else f"Goodbye {name.split()[0]}", f"Checked {log_type}!"]
            GPIO.output(GREEN_LED_PIN if log_type == "Check-In" else RED_LED_PIN, GPIO.HIGH)
            display_message(msg)
            time.sleep(3)
            GPIO.output(GREEN_LED_PIN if log_type == "Check-In" else RED_LED_PIN, GPIO.LOW)
        else:
            display_message(["User not", "recognized"])
    else:
        display_message(["Face not", "recognized"])
    time.sleep(2)

def monitor_buttons():
    debounce = 0.2
    last_in, last_out = time.time(), time.time()
    while True:
        if GPIO.input(CHECK_IN_PIN) == GPIO.LOW and time.time() - last_in > debounce:
            process_attendance("Check-In")
            last_in = time.time()
        elif GPIO.input(CHECK_OUT_PIN) == GPIO.LOW and time.time() - last_out > debounce:
            process_attendance("Check-Out")
            last_out = time.time()
        time.sleep(0.1)

def calculate_hours_worked():
    log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'], unit='s')
    log_df['Week'] = log_df['Timestamp'].dt.strftime('%Y-%U')
    summary = []
    for (name, week), group in log_df.groupby(['Name', 'Week']):
        group = group.sort_values('Timestamp')
        ins = group[group['Type'] == 'Check-In']['Timestamp'].tolist()
        outs = group[group['Type'] == 'Check-Out']['Timestamp'].tolist()
        total = sum((out - ins[i]).total_seconds() for i, out in enumerate(outs[:len(ins)]))
        summary.append({"Name": name, "Week": week, "Hours Worked": round(total / 3600, 2)})
    df = pd.DataFrame(summary)
    df.to_excel(WEEKLY_LOG_FILE, index=False)
    pg_cursor.execute("DELETE FROM weekly_hours")
    for row in df.itertuples():
        pg_cursor.execute("INSERT INTO weekly_hours (name, week, hours_worked) VALUES (%s, %s, %s)", (row.Name, row.Week, row._3))

# Scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(update_weekly_monthly_logs, 'cron', hour=20)
scheduler.add_job(update_weekly_monthly_logs, 'cron', day_of_week='fri', hour=22)
scheduler.start()

# Threads
threading.Thread(target=background_employee_sync, daemon=True).start()
threading.Thread(target=background_offline_sync, daemon=True).start()

# Main
if __name__ == '__main__':
    display_message(["System Ready", "Press Button"])
    try:
        monitor_buttons()
    except KeyboardInterrupt:
        lcd.clear()
        GPIO.cleanup()
        picam2.stop()
        calculate_hours_worked()
        pg_cursor.close()
        pg_conn.close()
        print("[SYSTEM] Shutdown")
