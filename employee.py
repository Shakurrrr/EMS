import RPi.GPIO as GPIO
import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import pickle
import time
from picamera2 import Picamera2
from RPLCD.i2c import CharLCD
import threading
import psycopg2
from config import POSTGRES_CONFIG
import requests
from io import BytesIO
from PIL import Image
from apscheduler.schedulers.background import BackgroundScheduler

# PostgreSQL Connection
pg_conn = psycopg2.connect(**POSTGRES_CONFIG)
pg_conn.autocommit = True
pg_cursor = pg_conn.cursor()

# Constants and File Paths
BASE_DIR = "/home/pi/Desktop/PROJECT/N-facial-recognition-QRCODE"
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pickle")
LOG_FILE = os.path.join(BASE_DIR, "employee_attendance_log.xlsx")
WEEKLY_LOG_FILE = os.path.join(BASE_DIR, "weekly_log.xlsx")
MONTHLY_LOG_FILE = os.path.join(BASE_DIR, "monthly_log.xlsx")
UNSYNCED_FILE = os.path.join(BASE_DIR, "unsynced_records.xlsx")

# GPIO Pins
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

try:
    log_df = pd.read_excel(LOG_FILE)
except FileNotFoundError:
    log_df = pd.DataFrame(columns=["Name", "EmployeeID", "Type", "Date", "Time", "Timestamp"])

# FSM placeholder
def transition_to(state):
    global STATE
    STATE = state
    print(f"[FSM] Transitioning to {state}")

def fetch_and_update_employees():
    global known_face_encodings, known_face_names
    pg_cursor.execute("SELECT name, image_url FROM employees")
    rows = pg_cursor.fetchall()

    names = []
    encodings = []

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
            print(f"[ERROR] Failed to process image for {name}: {e}")

    known_face_encodings = encodings
    known_face_names = names

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)

    print(f"[SYNC] Employee encodings updated: {len(names)} employees")

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

    success_list = []
    for index, row in df_unsynced.iterrows():
        try:
            pg_cursor.execute(
                """
                INSERT INTO employee_attendance (name, employee_id, type, date, time, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (row["Name"], row["EmployeeID"], row["Type"], row["Date"], row["Time"], row["Timestamp"])
            )
            success_list.append(index)
        except Exception as e:
            print(f"[OFFLINE SYNC] Failed to upload record: {e}")

    if success_list:
        df_unsynced.drop(success_list, inplace=True)
        df_unsynced.to_excel(UNSYNCED_FILE, index=False)
        print(f"[OFFLINE SYNC] Uploaded {len(success_list)} records")

def background_offline_sync():
    while True:
        upload_unsynced_records()
        time.sleep(300)

def update_weekly_monthly_logs():
    log_df['Date'] = pd.to_datetime(log_df['Date'])
    log_df['Week'] = log_df['Date'].dt.strftime('%Y-%U')
    log_df['Month'] = log_df['Date'].dt.to_period('M')

    weekly_summary = log_df.groupby(['Name', 'EmployeeID', 'Week', 'Type']).size().unstack(fill_value=0).reset_index()
    monthly_summary = log_df.groupby(['Name', 'EmployeeID', 'Month', 'Type']).size().unstack(fill_value=0).reset_index()

    weekly_summary.to_excel(WEEKLY_LOG_FILE, index=False)
    monthly_summary.to_excel(MONTHLY_LOG_FILE, index=False)

def recognize_face(timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        frame = picam2.capture_array()
        resized = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if matches and len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    confidence = round((1 - face_distances[best_match_index]) * 100, 2)
                    return name, confidence
    return None, None

def log_attendance(name, log_type):
    global log_df
    pg_cursor.execute("SELECT employee_id FROM employees WHERE name = %s LIMIT 1", (name,))
    result = pg_cursor.fetchone()
    if not result:
        return False

    employee_id = result[0]
    now = datetime.now()
    timestamp = now.timestamp()

    record = {
        "Name": name,
        "EmployeeID": employee_id,
        "Type": log_type,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Timestamp": timestamp
    }

    if name in recent_logs and (datetime.now() - recent_logs[name]).total_seconds() < 900:
        return "duplicate"

    log_df = pd.concat([log_df, pd.DataFrame([record])], ignore_index=True)
    log_df.to_excel(LOG_FILE, index=False)
    update_weekly_monthly_logs()

    try:
        pg_cursor.execute(
            """
            INSERT INTO employee_attendance (name, employee_id, type, date, time, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (record["Name"], record["EmployeeID"], record["Type"], record["Date"], record["Time"], record["Timestamp"])
        )
    except Exception as e:
        print(f"[OFFLINE MODE] Failed to upload, saving locally: {e}")
        save_unsynced_record(record)

    recent_logs[name] = datetime.now()
    return record

def display_message(lines, scroll_delay=0.3):
    lcd.clear()
    for line in lines:
        if len(line) <= 16:
            lcd.write_string(line.ljust(16))
            lcd.crlf()
        else:
            for i in range(len(line) - 15):
                lcd.clear()
                lcd.write_string(line[i:i+16])
                lcd.crlf()
                time.sleep(scroll_delay)
            time.sleep(1)

def process_attendance(log_type):
    display_message(["Look at the camera"])
    name, confidence = recognize_face()
    if name:
        first_name = name.split()[0]
        display_message([f"Detected: {name}", f"Match: {confidence}%"])
        time.sleep(1.5)
        result = log_attendance(name, log_type)
        if result == "duplicate":
            display_message(["Already", f"Checked {log_type}".lower()])
        elif result:
            if log_type == "Check-In":
                GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
                display_message([f"Welcome {first_name}", "Checked In!"])
                time.sleep(3)
                GPIO.output(GREEN_LED_PIN, GPIO.LOW)
            else:
                GPIO.output(RED_LED_PIN, GPIO.HIGH)
                display_message([f"Goodbye {first_name}", "Checked Out!"])
                time.sleep(3)
                GPIO.output(RED_LED_PIN, GPIO.LOW)
        else:
            display_message(["User not", "recognized"])
    else:
        display_message(["Face not", "recognized"])
    time.sleep(2)

def main():
    global STATE, pg_conn, pg_cursor

    transition_to("INIT")

    scheduler = BackgroundScheduler()
    scheduler.add_job(update_weekly_monthly_logs, 'cron', hour=20, minute=0)
    scheduler.add_job(update_weekly_monthly_logs, 'cron', day_of_week='fri', hour=22, minute=0)
    scheduler.start()

    display_message(["System Booting..."])
    time.sleep(3)

    sync_thread = threading.Thread(target=background_employee_sync, daemon=True)
    sync_thread.start()

    offline_sync_thread = threading.Thread(target=background_offline_sync, daemon=True)
    offline_sync_thread.start()

    display_message(["System Ready", "Press Button"])
    transition_to("IDLE")

    try:
        while True:
            if GPIO.input(CHECK_IN_PIN) == GPIO.LOW:
                transition_to("RECOGNIZING")
                process_attendance("Check-In")
                transition_to("IDLE")
            elif GPIO.input(CHECK_OUT_PIN) == GPIO.LOW:
                transition_to("RECOGNIZING")
                process_attendance("Check-Out")
                transition_to("IDLE")
            time.sleep(0.1)
    except KeyboardInterrupt:
        transition_to("SHUTDOWN")
        lcd.clear()
        GPIO.cleanup()
        picam2.stop()
        try:
            if pg_cursor: pg_cursor.close()
            if pg_conn: pg_conn.close()
        except Exception as e:
            print(f"[DB] Error closing PostgreSQL connection: {e}")
        print("[SYSTEM] Shutdown")

if __name__ == "__main__":
    main()
