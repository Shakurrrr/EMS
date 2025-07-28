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
from threading import Lock
lcd_lock = Lock()
last_greeting = None


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

BASE_DIR = "/home/pi/Desktop/PROJECT/EMSAT_EMPLOYEE/ATS_PROJECT"
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
            if self.pg_cursor:
                self.pg_cursor.close()
            if self.pg_conn:
                self.pg_conn.close()
        except Exception:
            pass 
    
        try:
            self.pg_conn = psycopg2.connect(**POSTGRES_CONFIG)
            self.pg_conn.autocommit = True
            self.pg_cursor = self.pg_conn.cursor()
            if not self.db_online:
                self.db_online = True
                print("[DB INIT] Connected to PostgreSQL")
                self.sync_offline_logs()
                display(["DB CONNECTED", "SYNCED LOGS"])
                time.sleep(3)
        except Exception as e:
            if self.db_online:
                print(f"[DB ERROR] Lost connection: {e}")
                dsplay(["DB OFFLINE", "LOG QUEUED"])
            self.db_online = False

    def retry_connection_loop(self):
        while True:
            if not self.db_online:
                self.check_db_connection()
            time.sleep(10)

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
                    if buffer.tell() > 3_000_000:
                        print(f"[SKIPPED] {name}: Image still exceeds 3MB after compression")
                        continue

                    arr = np.array(img)
                    locs = face_recognition.face_locations(arr)
                    print(f"[DEBUG] Processing {name}, URL: {url}, Face locations: {locs}")
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
                        INSERT INTO attendances (employee_id, type, attendance_date, time, timestamp, synced)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (eid, log["type"], log["date"], log["time"], log["timestamp"], True))
            os.remove(OFFLINE_LOG_FILE)
            print("[SYNC] Offline logs successfully synced")
        except Exception as e:
            print(f"[SYNC ERROR] {e}")

    def log_attendance(self, name, log_type):
        now = datetime.now()

        if not self.db_online:
            print("[ATTENDANCE] DB offline; queuing log")
            display(["DB OFFLINE:", "QUEUED"])
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
                WHERE LOWER(CONCAT(first_name, ' ', last_name)) = LOWER(%s)

            """, (name,))
            res = self.pg_cursor.fetchone()
            if not res:
                print(f"[ERROR] No matching employee found for name: {name}")
                return False
            eid = res[0]
            
            method_used = "facial_recognition"
            if log_type == "Check-In":
                self.pg_cursor.execute("""
                   SELECT id FROM attendances
                   WHERE employee_id = %s AND attendance_date = %s
                """, (eid, now.date()))
                if self.pg_cursor.fetchone():
                    display(["ALREADY CHECKED", "IN TODAY"])
                    print(f"[DUPLICATE] Check-in attempt for {name} already exists today.")
                    return "duplicate"

                clock_in_time = datetime.utcnow()
                self.pg_cursor.execute("""
                    INSERT INTO attendances (employee_id, attendance_date, clock_in, method, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, NOW(), NOW())
                """, (eid, now.date(), clock_in_time, method_used))
                
                # Mark late if after 9:00 AM UTC
                if clock_in_time.time() > datetime.strptime("09:00:00", "%H:%M:%S").time():
                    self.pg_cursor.execute("""
                        UPDATE attendances
                        SET is_late = TRUE
                        WHERE employee_id = %s AND attendance_date = %s
                    """, (eid, now.date()))
                
            #check out function
            elif log_type == "Check-Out":
                print(f"[INFO] Attempting Check-Out for {name}")
                clock_out_time = datetime.utcnow()
                
                # locating a record to update
                self.pg_cursor.execute("""
                    SELECT id FROM attendances 
                    WHERE employee_id = %s AND attendance_date = %s AND clock_out IS NULL
                    ORDER BY id DESC LIMIT 1
                """, (eid, now.date()))
                open_record = self.pg_cursor.fetchone()
                if not open_record:
                    display(["ALREADY", "CHECKED OUT"])
                    print("[ATTENDANCE] No open check-in found; clock-out failed.")
                    return "duplicate"
                print(f"[DEBUG] Targeting attendance ID {open_record[0]} for Check-Out")
                
                #clock out and total hours worked calculation
                self.pg_cursor.execute("""
                    UPDATE attendances
                    SET clock_out = %s,
                        total_hours = EXTRACT(EPOCH FROM (%s - clock_in)) / 3600.0,
                        updated_at = NOW()
                    WHERE employee_id = %s AND attendance_date = %s AND clock_out IS NULL
                """, (clock_out_time, clock_out_time, eid, now.date()))
                print(f"Rows updated for clock_out: {self.pg_cursor.rowcount}")
                print(f"[DEBUG] UPDATE ROWS AFFECTED: {self.pg_cursor.rowcount}")
                
                if self.pg_cursor.rowcount == 0:
                    print("[ATTENDANCE] No open check-in found; clock-out failed.")
                    display(["NO ACTIVE", "CHECK-IN FOUND"])
                    return False
                
            print(f"[LOGGED] {log_type} for {name} (employee_id={eid})")
            recent_logs[name] = now
            return True
        
            print("[ERROR] Unknown log_type passed.")
            return False


        except Exception as e:
            print(f"[ATTENDANCE ERROR] {e}")
            return False


    def export_attendance(self):
        if not self.db_online:
            print("[EXPORT] Failed: DB offline")
            return
        try:
            df = pd.read_sql_query("""
                SELECT e.first_name || ' ' || e.last_name as name,
                       a.employee_id,
                       a.type,
                       a.attendance_date as date,
                       a.time,
                       to_timestamp(a.timestamp) as datetime
                FROM attendances a
                JOIN employees e ON a.employee_id = e.id
                ORDER BY a.timestamp DESC
            """, self.pg_conn)
            df.to_excel(EXPORT_FILE, index=False)
            print(f"[EXPORT] Saved to {EXPORT_FILE}")
        except Exception as e:
            print(f"[EXPORT ERROR] {e}")



    def calculate_hours(self, start_date, end_date):
        if not self.db_online:
            return
        try:
            self.pg_cursor.execute("""
                SELECT employee_id, clock_in, clock_out, attendance_date 
                FROM attendances
                WHERE attendance_date BETWEEN %s AND %s
                ORDER BY employee_id, attendance_date
            """, (start_date, end_date))
            rows = self.pg_cursor.fetchall()
            work_hours = {}
            for emp_id, clock_in, clock_out, dt in rows:
                if emp_id not in work_hours:
                    work_hours[emp_id] = []
                work_hours[emp_id].append((clock_in, clock_out, dt))
            summaries = []
            for eid, logs in work_hours.items():
                logs_by_day = {}
                for clock_in, clock_out, dt in logs:
                    logs_by_day.setdefault(dt, []).append((clock_in, clock_out))
                total_seconds = 0
                for day, entries in logs_by_day.items():
                    for clock_in, clock_out in entries:
                        if clock_in and clock_out:
                            total_seconds += clock_out - clock_in
                summaries.append((eid, total_seconds))
            return summaries
        except Exception as e:
            print(f"[HOURS ERROR] {e}")


    def update_weekly_hours(self, reference_date=None):
        if not self.db_online:
            return
        today = datetime.now().date()
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        summary = self.calculate_hours(start, end)
        if summary:
            for emp_id, secs in summary:
                week_number = start.isocalendar()[1]
                year = start.year
                self.pg_cursor.execute("""
                    INSERT INTO weekly_attendance_logs (employee_id, week, year, total_minutes)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (employee_id, week, year) DO UPDATE
                    SET total_minutes = EXCLUDED.total_minutes
                """, (emp_id, week_number, year, secs // 60))

    def update_monthly_hours(self, reference_date=None):
        if not self.db_online:
            return
        today = datetime.now().date()
        start = today.replace(day=1)
        next_month = (start + timedelta(days=32)).replace(day=1)
        end = next_month - timedelta(days=1)
        summary = self.calculate_hours(start, end)
        if summary:
            for emp_id, secs in summary:
               month = start.month
               year = start.year
               self.pg_cursor.execute("""
                  INSERT INTO monthly_attendance_logs (employee_id, month, year, total_minutes)
                  VALUES (%s, %s, %s, %s)
                  ON CONFLICT (employee_id, month, year) DO UPDATE
                  SET total_minutes = EXCLUDED.total_minutes
               """, (emp_id, month, year, secs // 60))
               
    def mark_daily_absentees(self, target_date=None):
        if not self.db_online:
            return
        today = datetime.now().date() if target_date is None else target_date
        
        if today.weekday() >= 5:  # Skip Saturday and Sunday
            print(f"[ABSENT] Skipped absentee marking for weekend ({today})")
            return
        
        try:
            # All active employee IDs
            self.pg_cursor.execute("SELECT id FROM employees")
            all_employees = set(row[0] for row in self.pg_cursor.fetchall())
            
            # Those who logged attendance today
            self.pg_cursor.execute("""
                SELECT DISTINCT employee_id
                FROM attendances
                WHERE attendance_date = %s
            """, (today,))
            present_employees = set(row[0] for row in self.pg_cursor.fetchall())
            
            # Absent ones = all - present
            absent_employees = all_employees - present_employees

            for eid in absent_employees:
                self.pg_cursor.execute("""
                    INSERT INTO attendances (employee_id, attendance_date, is_absent, created_at, updated_at)
                    VALUES (%s, %s, TRUE, NOW(), NOW())
                """, (eid, today))
                
            print(f"[ABSENT] Marked {len(absent_employees)} employees absent for {today}")
        except Exception as e:
            print(f"[ABSENT ERROR] {e}")

    
    

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

def recognize_face(timeout=15, headless=True):
    start = time.time()
    while time.time() - start < timeout:
        frame = picam2.capture_array()
        resized = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)

        if not headless:
            for (top, right, bottom, left) in locs:
                cv2.rectangle(resized, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imshow("Camera Preview", resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for enc in encs:
            matches = face_recognition.compare_faces(known_face_encodings, enc)
            dists = face_recognition.face_distance(known_face_encodings, enc)
            if matches and dists.size > 0:
                idx = np.argmin(dists)
                if matches[idx]:
                    if not headless:
                        cv2.destroyAllWindows()
                    return known_face_names[idx], round((1 - dists[idx]) * 100, 2)

    if not headless:
        cv2.destroyAllWindows()
    return None, None

def display(lines, delay=0.3):
    with lcd_lock:
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
            

def hourly_greeting_updater():
    last_hour = -1
    while True:
        current_hour = datetime.now().hour
        if current_hour != last_hour:
            display(get_greeting_lines())
            last_hour = current_hour
        time.sleep(30)  # Poll every 30 seconds
       
            
def get_greeting_lines():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return ["GOOD MORNING!", "PRESS TO LOG"]
    elif 12 <= hour < 17:
        return ["GOOD AFTERNOON!", "PRESS TO LOG"]
    elif 17 <= hour < 21:
        return ["GOOD EVENING!", "PRESS TO LOG"]
    else:
        return ["WELCOME BACK!", "PRESS TO LOG"]
                    

def handle_attendance(mgr, action, headless=True):
    print(f"[DEBUG] Button triggered action: {action}")
    display(["LOOK AT THE CAMERA"])
    name, confidence = recognize_face(headless=headless)
    if name:
        display([f"DETECTED: {name.upper()}", f"MATCH: {confidence}%"])
        time.sleep(1.5)
        result = mgr.log_attendance(name, action)
        if result == "duplicate":
            display(["ALREADY", f"CHECKED {action}".upper()])
            time.sleep(2)
            display(get_greeting_lines())
            transition_to("IDLE") 
            print(f"[SKIP] Duplicate {action} attempt for {name}")
            return
        
        elif result:
            GPIO.output(GREEN_LED_PIN if action == "Check-In" else RED_LED_PIN, GPIO.HIGH)
            print(f"[LED] Lighting up {'GREEN' if action == 'Check-In' else 'RED'} for {action}")
            display([f"{name.split()[0].upper()} {action.upper()}", "SUCCESS!"])
            time.sleep(3)
            GPIO.output(GREEN_LED_PIN if action == "Check-In" else RED_LED_PIN, GPIO.LOW)
        else:
            display(["USER NOT", "RECOGNIZED"])
    else:
        display(["FACE NOT", "RECOGNIZED"])
    time.sleep(2)

    display(get_greeting_lines())
    transition_to("IDLE")

    
    ##system FSM

def main():
    mgr = AttendanceManager()
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)

    # Only mark absentees for workdays
    if yesterday.weekday() < 5:  # 0 = Monday, 6 = Sunday
        try:
            mgr.pg_cursor.execute("""
                SELECT COUNT(*) FROM attendances
                WHERE attendance_date = %s AND is_absent = TRUE
            """, (yesterday,))
            if mgr.pg_cursor.fetchone()[0] == 0:
                mgr.mark_daily_absentees(yesterday)
        except Exception as e:
            print(f"[ABSENT FALLBACK ERROR] {e}")

    # Weekly check fallback (if Monday)
    if today.weekday() == 0:  # Monday
        last_friday = today - timedelta(days=3)
        try:
            week_num = last_friday.isocalendar()[1]
            mgr.pg_cursor.execute("""
                SELECT COUNT(*) FROM weekly_attendance_logs
                WHERE week = %s AND year = %s
            """, (week_num, last_friday.year))
            if mgr.pg_cursor.fetchone()[0] == 0:
                mgr.update_weekly_hours(last_friday)
        except Exception as e:
            print(f"[WEEKLY FALLBACK ERROR] {e}")

    # Monthly check fallback (if 1st of month)
    if today.day == 1:
        last_month = today.replace(day=1) - timedelta(days=1)
        try:
            mgr.pg_cursor.execute("""
                SELECT COUNT(*) FROM monthly_attendance_logs
                WHERE month = %s AND year = %s
            """, (last_month.month, last_month.year))
            if mgr.pg_cursor.fetchone()[0] == 0:
                mgr.update_monthly_hours(last_month)
        except Exception as e:
            print(f"[MONTHLY FALLBACK ERROR] {e}")
    scheduler = BackgroundScheduler()
    scheduler.add_job(mgr.fetch_and_update_employees, 'interval', minutes=5)
    scheduler.add_job(mgr.update_weekly_hours, 'cron', day_of_week='fri', hour=22, minute=0)
    scheduler.add_job(mgr.update_monthly_hours, 'cron', day='last', hour=22, minute=0)
    scheduler.add_job(mgr.mark_daily_absentees, 'cron', hour=23, minute=59)
    scheduler.start()

    threading.Thread(target=mgr.fetch_and_update_employees, daemon=True).start()
    threading.Thread(target=hourly_greeting_updater, daemon=True).start()


    display(get_greeting_lines())
    transition_to("IDLE")
    try:
        while True:
            if GPIO.input(CHECK_IN_PIN) == GPIO.LOW:
                transition_to("RECOGNIZING")
                handle_attendance(mgr, "Check-In")
                transition_to("IDLE")
            elif GPIO.input(CHECK_OUT_PIN) == GPIO.LOW:
                print("Check-Out button pressed")
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


