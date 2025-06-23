# EMSATS Attendance System

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

# Postgres connection
pg_conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="emsats_db",
    user="emsats_user",
    password="mypassword123"
)
pg_conn.autocommit = True
pg_cursor = pg_conn.cursor()

# Paths & Constants
BASE_DIR = "/home/pi/Desktop/PROJECT/N-facial-recognition-QRCODE"
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pickle")
LOG_FILE = os.path.join(BASE_DIR, "employee_attendance_log.xlsx")
WEEKLY_LOG_FILE = os.path.join(BASE_DIR, "weekly_log.xlsx")
MONTHLY_LOG_FILE = os.path.join(BASE_DIR, "monthly_log.xlsx")
UNSYNCED_FILE = os.path.join(BASE_DIR, "unsynced_records.xlsx")

# GPIO Setup
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

# Load attendance log or initialize
try:
    log_df = pd.read_excel(LOG_FILE)
except FileNotFoundError:
    log_df = pd.DataFrame(columns=["Name", "EmployeeID", "Type", "Date", "Time", "Timestamp"])

# Fetch and Update Employee Encodings

def fetch_and_update_employees():
    global known_face_encodings, known_face_names

    try:
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
                print(f"[ERROR] Failed to process image for {name}: {e}")

        known_face_encodings = encodings
        known_face_names = names

        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)

        print(f"[SYNC] Employee encodings updated: {len(names)} employees")

    except Exception as e:
        print(f"[SYNC ERROR] Could not update employee encodings: {e}")


def background_employee_sync():
    while True:
        fetch_and_update_employees()
        time.sleep(300)  # Every 5 minutes


# Offline Sync

def save_unsynced_record(record):
    try:
        df_unsynced = pd.read_excel(UNSYNCED_FILE)
    except FileNotFoundError:
        df_unsynced = pd.DataFrame(columns=["Name", "EmployeeID", "Type", "Date", "Time", "Timestamp"])

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
        time.sleep(300)  # Every 5 minutes


# Weekly / Monthly Logs

def update_weekly_monthly_logs():
    log_df['Date'] = pd.to_datetime(log_df['Date'])
    log_df['Week'] = log_df['Date'].dt.strftime('%Y-%U')
    log_df['Month'] = log_df['Date'].dt.to_period('M')

    weekly_summary = log_df.groupby(['Name', 'EmployeeID', 'Week', 'Type']).size().unstack(fill_value=0).reset_index()
    monthly_summary = log_df.groupby(['Name', 'EmployeeID', 'Month', 'Type']).size().unstack(fill_value=0).reset_index()

    # Save to Excel
    weekly_summary.to_excel(WEEKLY_LOG_FILE, index=False)
    monthly_summary.to_excel(MONTHLY_LOG_FILE, index=False)

    # Update PostgreSQL
    try:
        pg_cursor.execute("DELETE FROM weekly_reports")
        for _, row in weekly_summary.iterrows():
            pg_cursor.execute(
                """
                INSERT INTO weekly_reports (name, employee_id, week, check_in, check_out)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (row["Name"], row["EmployeeID"], row["Week"], row.get("Check-In", 0), row.get("Check-Out", 0))
            )

        pg_cursor.execute("DELETE FROM monthly_reports")
        for _, row in monthly_summary.iterrows():
            pg_cursor.execute(
                """
                INSERT INTO monthly_reports (name, employee_id, month, check_in, check_out)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (row["Name"], row["EmployeeID"], str(row["Month"]), row.get("Check-In", 0), row.get("Check-Out", 0))
            )

        print(f"[REPORT] Weekly & Monthly logs updated at {datetime.now()}")

    except Exception as e:
        print(f"[REPORT ERROR] Failed to update reports: {e}")

# Face Recognition

def recognize_face(timeout=2, headless=True):
    start = time.time()
    face_names = []

    while time.time() - start < timeout:
        frame = picam2.capture_array()

        resized = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            name = "Unknown"

            if matches and len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    confidence = round((1 - face_distances[best_match_index]) * 100, 2)

                    if headless:
                        return name, confidence

            face_names.append(name)

        # Display window (non-headless)
        if not headless:
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= cv_scaler; right *= cv_scaler; bottom *= cv_scaler; left *= cv_scaler
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if not headless:
        cv2.destroyAllWindows()

    return None, None


# Log Attendance

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

    # Duplicate prevention (within 15 mins)
    if name in recent_logs and (datetime.now() - recent_logs[name]).total_seconds() < 900:
        return "duplicate"

    # Save to log
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


# LCD Display

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


# Attendance Process

def process_attendance(log_type):
    display_message(["Look at the camera"])
    name, confidence = recognize_face(timeout=15, headless=True)

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


# Button Monitoring

def monitor_buttons():
    debounce = 0.2
    last_check_in = time.time()
    last_check_out = time.time()

    while True:
        if GPIO.input(CHECK_IN_PIN) == GPIO.LOW and time.time() - last_check_in > debounce:
            process_attendance("Check-In")
            last_check_in = time.time()
        elif GPIO.input(CHECK_OUT_PIN) == GPIO.LOW and time.time() - last_check_out > debounce:
            process_attendance("Check-Out")
            last_check_out = time.time()

        time.sleep(0.1)


# Calculate Weekly Hours

def calculate_hours_worked():
    log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'], unit='s')
    log_df['Week'] = log_df['Timestamp'].dt.strftime('%Y-%U')

    weekly_hours = []

    for (name, week), group in log_df.groupby(['Name', 'Week']):
        group = group.sort_values('Timestamp')
        check_ins = group[group['Type'] == 'Check-In']['Timestamp'].tolist()
        check_outs = group[group['Type'] == 'Check-Out']['Timestamp'].tolist()

        total_seconds = 0
        for in_time, out_time in zip(check_ins, check_outs):
            delta = out_time - in_time
            total_seconds += delta.total_seconds()

        hours_worked = total_seconds / 3600
        weekly_hours.append({"Name": name, "Week": week, "Hours Worked": round(hours_worked, 2)})

    weekly_hours_df = pd.DataFrame(weekly_hours)
    weekly_hours_df.to_excel(WEEKLY_LOG_FILE, index=False)

    try:
        pg_cursor.execute("DELETE FROM weekly_hours")
        for _, row in weekly_hours_df.iterrows():
            pg_cursor.execute(
                """
                INSERT INTO weekly_hours (name, week, hours_worked)
                VALUES (%s, %s, %s)
                """,
                (row["Name"], row["Week"], row["Hours Worked"])
            )
        print(f"[HOURS] Weekly hours updated at {datetime.now()}")

    except Exception as e:
        print(f"[HOURS ERROR] Failed to update weekly hours: {e}")

# Scheduler Setup

scheduler = BackgroundScheduler()

# Daily report at 8:00 PM
scheduler.add_job(update_weekly_monthly_logs, 'cron', hour=20, minute=0)
print("[SCHEDULER] Daily report update scheduled for 8:00 PM")

# Weekly report every Friday at 10:00 PM
scheduler.add_job(update_weekly_monthly_logs, 'cron', day_of_week='fri', hour=22, minute=0)
print("[SCHEDULER] Weekly report scheduled for Fridays at 10:00 PM")

# Start Scheduler
scheduler.start()


# Background Threads

# Employee sync thread
sync_thread = threading.Thread(target=background_employee_sync, daemon=True)
sync_thread.start()

# Offline sync thread
offline_sync_thread = threading.Thread(target=background_offline_sync, daemon=True)
offline_sync_thread.start()

print("[SYSTEM] Scheduler and sync threads started. System is running.")


# Main Program

display_message(["System Ready", "Press Button"])

try:
    monitor_buttons()

except KeyboardInterrupt:
    print("[SYSTEM] Shutdown initiated by user...")
    lcd.clear()
    GPIO.cleanup()
    picam2.stop()

    calculate_hours_worked()

    pg_cursor.close()
    pg_conn.close()

    print("[SYSTEM] Shutdown complete.")

