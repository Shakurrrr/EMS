import face_recognition
import cv2
import numpy as np
from datetime import datetime
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

# PostgreSQL setup
pg_conn = psycopg2.connect(**POSTGRES_CONFIG)
pg_conn.autocommit = True
pg_cursor = pg_conn.cursor()

# Constants
BASE_DIR = "/home/pi/Desktop/PROJECT/N-facial-recognition-QRCODE"
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pickle")

# GPIO mock for dev
try:
    import RPi.GPIO as GPIO
except ImportError:
    class GPIO:
        BCM = IN = OUT = LOW = HIGH = PUD_UP = 0
        @staticmethod
        def setmode(*args): pass
        @staticmethod
        def setup(*args, **kwargs): pass
        @staticmethod
        def input(pin): return 1
        @staticmethod
        def output(pin, value): pass
        @staticmethod
        def cleanup(): pass

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

# LCD Setup
lcd = CharLCD('PCF8574', 0x27)

# Camera Setup
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Globals
cv_scaler = 2
recent_logs = {}
known_face_encodings = []
known_face_names = []
unsynced_memory = []

# FSM
STATE = "INIT"
def transition_to(state):
    global STATE
    STATE = state
    print(f"[FSM] Transition to {state}")

# Sync Employee Photos
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
            face_encs = face_recognition.face_encodings(img_np, face_locations)
            if face_encs:
                names.append(name)
                encodings.append(face_encs[0])
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
    known_face_encodings = encodings
    known_face_names = names
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
    print(f"[SYNC] Encoded {len(names)} employees")

def background_employee_sync():
    while True:
        fetch_and_update_employees()
        time.sleep(300)

# Offline Sync
def upload_unsynced():
    global unsynced_memory
    remaining = []
    for r in unsynced_memory:
        try:
            pg_cursor.execute("""
                INSERT INTO employee_attendance (name, employee_id, type, date, time, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (r["Name"], r["EmployeeID"], r["Type"], r["Date"], r["Time"], r["Timestamp"]))
        except Exception as e:
            print(f"[RETRY FAILED] {e}")
            remaining.append(r)
    unsynced_memory = remaining

def background_offline_sync():
    while True:
        upload_unsynced()
        time.sleep(300)

# Face Recognition
def recognize_face(timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        frame = picam2.capture_array()
        resized = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for enc in encodings:
            matches = face_recognition.compare_faces(known_face_encodings, enc)
            distances = face_recognition.face_distance(known_face_encodings, enc)
            if matches and distances.size > 0:
                best_idx = np.argmin(distances)
                if matches[best_idx]:
                    return known_face_names[best_idx], round((1 - distances[best_idx]) * 100, 2)
    return None, None

# Logging Attendance
def log_attendance(name, log_type):
    pg_cursor.execute("SELECT employee_id FROM employees WHERE name = %s LIMIT 1", (name,))
    result = pg_cursor.fetchone()
    if not result:
        return False
    emp_id = result[0]
    now = datetime.now()
    record = {
        "Name": name,
        "EmployeeID": emp_id,
        "Type": log_type,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Timestamp": now.timestamp()
    }
    if name in recent_logs and (now - recent_logs[name]).total_seconds() < 900:
        return "duplicate"
    try:
        pg_cursor.execute("""
            INSERT INTO employee_attendance (name, employee_id, type, date, time, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (record["Name"], record["EmployeeID"], record["Type"], record["Date"], record["Time"], record["Timestamp"]))
    except Exception as e:
        print(f"[CACHE MODE] {e}")
        unsynced_memory.append(record)
    recent_logs[name] = now
    return record

# LCD Display
def display_message(lines, delay=0.3):
    lcd.clear()
    for line in lines:
        if len(line) <= 16:
            lcd.write_string(line.ljust(16))
            lcd.crlf()
        else:
            for i in range(len(line)-15):
                lcd.clear()
                lcd.write_string(line[i:i+16])
                lcd.crlf()
                time.sleep(delay)
            time.sleep(1)

# Process Check-in/out
def process_attendance(log_type):
    display_message(["Look at the camera"])
    name, confidence = recognize_face()
    if name:
        first = name.split()[0]
        display_message([f"Detected: {name}", f"Match: {confidence}%"])
        time.sleep(1.5)
        result = log_attendance(name, log_type)
        if result == "duplicate":
            display_message(["Already", f"Checked {log_type}".lower()])
        elif result:
            GPIO.output(GREEN_LED_PIN if log_type == "Check-In" else RED_LED_PIN, GPIO.HIGH)
            display_message([f"{first} {log_type}", "Success!"])
            time.sleep(3)
            GPIO.output(GREEN_LED_PIN if log_type == "Check-In" else RED_LED_PIN, GPIO.LOW)
        else:
            display_message(["User not", "recognized"])
    else:
        display_message(["Face not", "recognized"])
    time.sleep(2)

# Main App
def main():
    transition_to("INIT")
    BackgroundScheduler().start()
    threading.Thread(target=background_employee_sync, daemon=True).start()
    threading.Thread(target=background_offline_sync, daemon=True).start()
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
        pg_cursor.close()
        pg_conn.close()
        print("[SYSTEM] Shutdown")

if __name__ == "__main__":
    main()