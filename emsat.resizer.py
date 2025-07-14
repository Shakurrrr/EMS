import os
import time
import pickle
import threading
import requests
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

# Globals
cv_scaler = 2
recent_logs = {}
known_face_encodings = []
known_face_names = []

## handling database attendance

class AttendanceManager:
    def __init__(self):
        try:
            self.pg_conn = psycopg2.connect(**POSTGRES_CONFIG)
            self.pg_conn.autocommit = True
            self.pg_cursor = self.pg_conn.cursor()
            self.db_online = True
        except Exception as e:
            print(f"[DB INIT] PostgreSQL connection failed: {e}")
            self.pg_conn = None
            self.pg_cursor = None
            self.db_online = False

## fetcing data from the database

    def fetch_and_update_employees(self):
        global known_face_encodings, known_face_names
        if not self.db_online:
            print("[SYNC] Skipped: DB offline")
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

    def export_attendance(self):
        if not self.db_online:
            print("[EXPORT] Failed: DB offline")
            return
        try:
            df = pd.read_sql_query("""
                SELECT e.first_name || ' ' || e.last_name as name,
                       ea.employee_id,
                       ea.type,
                       ea.date,
                       ea.time,
                       to_timestamp(ea.timestamp) as datetime
                FROM employee_attendance ea
                JOIN employees e ON ea.employee_id = e.id
                ORDER BY ea.timestamp DESC
            """, self.pg_conn)
            df.to_excel(EXPORT_FILE, index=False)
            print(f"[EXPORT] Saved to {EXPORT_FILE}")
        except Exception as e:
            print(f"[EXPORT ERROR] {e}")

    def log_attendance(self, name, log_type):
        if not self.db_online:
            print("[ATTENDANCE] DB offline; skipping")
            return False
        try:
            self.pg_cursor.execute("""
                SELECT id FROM employees 
                WHERE CONCAT(first_name, ' ', last_name) = %s LIMIT 1
            """, (name,))
            res = self.pg_cursor.fetchone()
            if not res:
                return False
            eid = res[0]
            now = datetime.now()
            if name in recent_logs and (now - recent_logs[name]).total_seconds() < 900:
                return "duplicate"
            self.pg_cursor.execute("""
                INSERT INTO employee_attendance (employee_id, type, date, time, timestamp)
                VALUES (%s, %s, %s, %s, %s)
            """, (eid, log_type, now.date(), now.strftime("%H:%M:%S"), now.timestamp()))
            recent_logs[name] = now
            return True
        except Exception as e:
            print(f"[ATTENDANCE ERROR] {e}")
            return False

##   Calculated daily working hours    

    def calculate_hours(self, start_date, end_date):
        if not self.db_online:
            return
        try:
            self.pg_cursor.execute("""
                SELECT employee_id, type, timestamp, date FROM employee_attendance
                WHERE date BETWEEN %s AND %s
                ORDER BY employee_id, timestamp
            """, (start_date, end_date))
            rows = self.pg_cursor.fetchall()
            work_hours = {}
            for emp_id, typ, ts, dt in rows:
                if emp_id not in work_hours:
                    work_hours[emp_id] = []
                work_hours[emp_id].append((typ, ts, dt))
            summaries = []
            for eid, logs in work_hours.items():
                logs_by_day = {}
                for typ, ts, dt in logs:
                    logs_by_day.setdefault(dt, []).append((typ, ts))
                total_seconds = 0
                for day, entries in logs_by_day.items():
                    check_in = None
                    for typ, ts in entries:
                        if typ == 'Check-In':
                            check_in = ts
                        elif typ == 'Check-Out' and check_in:
                            total_seconds += ts - check_in
                            check_in = None
                summaries.append((eid, total_seconds))
            return summaries
        except Exception as e:
            print(f"[HOURS ERROR] {e}")

    ## updating weekly and monthly hours

    def update_weekly_hours(self):
        today = datetime.now().date()
        start = today - timedelta(days=today.weekday())  # Monday
        end = start + timedelta(days=6)  # Sunday
        summary = self.calculate_hours(start, end)
        if summary:
            for emp_id, secs in summary:
                self.pg_cursor.execute("""
                    INSERT INTO weekly_logs (employee_id, week_start, total_hours)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (employee_id, week_start) DO UPDATE
                    SET total_hours = EXCLUDED.total_hours
                """, (emp_id, start, secs // 3600))

    def update_monthly_hours(self):
        today = datetime.now().date()
        start = today.replace(day=1)
        next_month = (start + timedelta(days=32)).replace(day=1)
        end = next_month - timedelta(days=1)
        summary = self.calculate_hours(start, end)
        if summary:
            for emp_id, secs in summary:
                self.pg_cursor.execute("""
                    INSERT INTO monthly_logs (employee_id, month_start, total_hours)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (employee_id, month_start) DO UPDATE
                    SET total_hours = EXCLUDED.total_hours
                """, (emp_id, start, secs // 3600))

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

lcd = CharLCD('PCF8574', 0x27)
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

def transition_to(state):
    print(f"[FSM] → {state}")

def recognize_face(timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        frame = picam2.capture_array()
        resized = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        for enc in encs:
            matches = face_recognition.compare_faces(known_face_encodings, enc)
            dists = face_recognition.face_distance(known_face_encodings, enc)
            if matches and dists.size > 0:
                idx = np.argmin(dists)
                if matches[idx]:
                    return known_face_names[idx], round((1 - dists[idx]) * 100, 2)
    return None, None

def display(lines, delay=0.3):
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

def handle_attendance(mgr, action):
    display(["Look at the camera"])
    name, confidence = recognize_face()
    if name:
        display([f"Detected: {name}", f"Match: {confidence}%"])
        time.sleep(1.5)
        result = mgr.log_attendance(name, action)
        if result == "duplicate":
            display(["Already", f"Checked {action}".lower()])
        elif result:
            GPIO.output(GREEN_LED_PIN if action == "Check-In" else RED_LED_PIN, GPIO.HIGH)
            display([f"{name.split()[0]} {action}", "Success!"])
            time.sleep(3)
            GPIO.output(GREEN_LED_PIN if action == "Check-In" else RED_LED_PIN, GPIO.LOW)
        else:
            display(["User not", "recognized"])
    else:
        display(["Face not", "recognized"])
    time.sleep(2)

    ##system FSM

def main():
    mgr = AttendanceManager()
    scheduler = BackgroundScheduler()
    scheduler.add_job(mgr.fetch_and_update_employees, 'interval', minutes=5)
    scheduler.add_job(mgr.update_weekly_hours, 'cron', day_of_week='fri', hour=22, minute=0)
    scheduler.add_job(mgr.update_monthly_hours, 'cron', day='last', hour=22, minute=0)
    scheduler.start()

    threading.Thread(target=mgr.fetch_and_update_employees, daemon=True).start()

    display(["System Ready", "Press Button"])
    transition_to("IDLE")
    try:
        while True:
            if GPIO.input(CHECK_IN_PIN) == GPIO.LOW:
                transition_to("RECOGNIZING")
                handle_attendance(mgr, "Check-In")
                transition_to("IDLE")
            elif GPIO.input(CHECK_OUT_PIN) == GPIO.LOW:
                transition_to("RECOGNIZING")
                handle_attendance(mgr, "Check-Out")
                transition_to("IDLE")
            time.sleep(0.1)
    except KeyboardInterrupt:
        lcd.clear()
        GPIO.cleanup()
        picam2.stop()
        print("[SYSTEM] Shutdown")

if __name__ == "__main__":
    main()
