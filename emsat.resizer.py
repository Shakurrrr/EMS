import os
import time
import pickle
import threading
import requests
import json
from io import BytesIO
from datetime import datetime, timedelta
import RPi.GPIO as GPIO
import numpy as np
import pandas as pd
import cv2
import face_recognition
import psycopg2
from dotenv import load_dotenv
from PIL import Image, ImageFile
from picamera2 import Picamera2
from RPLCD.i2c import CharLCD
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Config
POSTGRES_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", 5432),
    "user": os.getenv("DB_USERNAME"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_DATABASE"),
}

BASE_DIR = "/home/pi/Desktop/PROJECT/N-facial-recognition-QRCODE"
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pickle")
EXPORT_FILE = os.path.join(BASE_DIR, "exported_attendance.xlsx")
OFFLINE_LOG_FILE = os.path.join(BASE_DIR, "offline_logs.json")

cv_scaler = 2
recent_logs = {}
known_face_encodings = []
known_face_names = []

class AttendanceManager:
    def __init__(self):
        self.db_online = False
        self.pg_conn = None
        self.pg_cursor = None
        self.check_db_connection()
        self.retry_thread = threading.Thread(target=self.retry_connection_loop, daemon=True)
        self.retry_thread.start()

    def check_db_connection(self):
        try:
            self.pg_conn = psycopg2.connect(**POSTGRES_CONFIG)
            self.pg_conn.autocommit = True
            self.pg_cursor = self.pg_conn.cursor()
            if not self.db_online:
                self.db_online = True
                print("[DB INIT] Connected to PostgreSQL")
                self.sync_offline_logs()
                display(["DB Connected", "Synced logs"])
        except Exception as e:
            if self.db_online:
                print(f"[DB ERROR] Lost connection: {e}")
            self.db_online = False

    def retry_connection_loop(self):
        while True:
            if not self.db_online:
                self.check_db_connection()
            time.sleep(10)  # Retry every 10 seconds

    def fetch_and_update_employees(self):
        global known_face_encodings, known_face_names
        if not self.db_online:
            print("[SYNC] DB offline: loading saved encodings...")
            if os.path.exists(ENCODINGS_FILE):
                with open(ENCODINGS_FILE, "rb") as f:
                    data = pickle.load(f)
                    known_face_encodings = data["encodings"]
                    known_face_names = data["names"]
                    print(f"[SYNC] Loaded {len(known_face_names)} faces from offline cache")
            else:
                print("[SYNC] No offline data available (encodings.pickle missing)")
            return

        try:
            self.pg_cursor.execute("SELECT id, first_name, last_name, photo FROM employees")
            rows = self.pg_cursor.fetchall()
            names, encodings = [], []
            for emp_id, first_name, last_name, url in rows:
                name = f"{first_name} {last_name}"
                try:
                    if not url:
                        continue
                    response = requests.get(url)
                    img = Image.open(BytesIO(response.content)).convert('RGB')

                    if img.width > 500:
                        img = img.resize((500, int(500 * img.height / img.width)))

                    buffer = BytesIO()
                    img.save(buffer, format='JPEG', quality=75)
                    if buffer.tell() > 2_000_000:
                        print(f"[SKIPPED] {name}: Image still exceeds 2MB after compression")
                        continue

                    arr = np.array(img)
                    locs = face_recognition.face_locations(arr)
                    encs = face_recognition.face_encodings(arr, locs)
                    if encs:
                        names.append(name)
                        encodings.append(encs[0])
                except Exception as err:
                    print(f"[ERROR] {name}: {err}")
            known_face_encodings = encodings
            known_face_names = names
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump({"encodings": encodings, "names": names}, f)
            print(f"[SYNC] Encoded {len(names)} employees")
        except Exception as e:
            print(f"[DB] Sync error: {e}")

    def sync_offline_logs(self):
        if not self.db_online or not os.path.exists(OFFLINE_LOG_FILE):
            return
        try:
            with open(OFFLINE_LOG_FILE, "r") as f:
                logs = json.load(f)
            for log in logs:
                self.pg_cursor.execute("""
                    SELECT id FROM employees 
                    WHERE CONCAT(first_name, ' ', last_name) = %s LIMIT 1
                """, (log["name"],))
                res = self.pg_cursor.fetchone()
                if res:
                    eid = res[0]
                    self.pg_cursor.execute("""
                        INSERT INTO employee_attendance (employee_id, type, date, time, timestamp)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (eid, log["type"], log["date"], log["time"], log["timestamp"]))
            os.remove(OFFLINE_LOG_FILE)
            print("[SYNC] Offline logs successfully synced")
        except Exception as e:
            print(f"[SYNC ERROR] {e}")

    def log_attendance(self, name, log_type):
        now = datetime.now()
        if name in recent_logs and (now - recent_logs[name]).total_seconds() < 900:
            return "duplicate"

        if not self.db_online:
            print("[ATTENDANCE] DB offline; queuing log")
            display(["DB Offline:", "Queued"])
            offline_log = {
                "name": name,
                "type": log_type,
                "date": str(now.date()),
                "time": now.strftime("%H:%M:%S"),
                "timestamp": now.timestamp()
            }
            try:
                logs = []
                if os.path.exists(OFFLINE_LOG_FILE):
                    with open(OFFLINE_LOG_FILE, "r") as f:
                        logs = json.load(f)
                logs.append(offline_log)
                with open(OFFLINE_LOG_FILE, "w") as f:
                    json.dump(logs, f, indent=4)
            except Exception as e:
                print(f"[QUEUE ERROR] Failed to queue log: {e}")
            return True

        try:
            self.pg_cursor.execute("""
                SELECT id FROM employees 
                WHERE CONCAT(first_name, ' ', last_name) = %s LIMIT 1
            """, (name,))
            res = self.pg_cursor.fetchone()
            if not res:
                return False
            eid = res[0]
            self.pg_cursor.execute("""
                INSERT INTO employee_attendance (employee_id, type, date, time, timestamp)
                VALUES (%s, %s, %s, %s, %s)
            """, (eid, log_type, now.date(), now.strftime("%H:%M:%S"), now.timestamp()))
            recent_logs[name] = now
            return True
        except Exception as e:
            print(f"[ATTENDANCE ERROR] {e}")
            return False

    # ... other methods remain unchanged ...
