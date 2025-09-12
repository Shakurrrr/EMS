# -*- coding: utf-8 -*-

import os
import time
import pickle
import threading
import json
from datetime import datetime, timedelta, timezone, date, time as dtime

import RPi.GPIO as GPIO
import numpy as np
import pandas as pd
import cv2
import face_recognition
from PIL import Image, ImageFile
from picamera2 import Picamera2
from RPLCD.i2c import CharLCD
from apscheduler.schedulers.background import BackgroundScheduler
from threading import Lock

# -------------------- Global Locks/State --------------------
lcd_lock = Lock()
encoding_ready = threading.Event()
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------- Base Paths --------------------
BASE_DIR = "/home/pi/Desktop/PROJECT/EMSAT_EMPLOYEE/ATS_PROJECT"

EMP_DIR = os.path.join(BASE_DIR, "employees")                 # employees/<First_Last>/photo_files
ENCODINGS_FILE = os.path.join(EMP_DIR, "encodings.pickle")    # cache of encodings
ATTEND_DIR = os.path.join(BASE_DIR, "attendance")             # daily CSVs
REPORTS_DIR = os.path.join(BASE_DIR, "reports")               # weekly/monthly CSV/PDF

os.makedirs(EMP_DIR, exist_ok=True)
os.makedirs(ATTEND_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# -------------------- Recognition Config --------------------
cv_scaler = 2
known_face_encodings = []
known_face_names = []          # full_name (spaces)
known_face_ids = []            # employee_id (folder name)
TOLERANCE = 0.45               # tune for your lighting/camera

# ---------- Output Policy ----------
WRITE_CSV = True               # MUST stay True; duplicates/open-check rely on CSV state
WRITE_DAILY_PDF = True         # auto-generate daily PDF after each log
DAILY_PDF_DIR = REPORTS_DIR    # where to store daily PDFs

# Columns to show in the daily PDF (keep CSV complete)
DAILY_PDF_COLUMNS = [
    "full_name", "attendance_date", "clock_in", "clock_out",
    "duration", "is_late", "is_absent"
]

# -------------------- Timezone/Business Rules --------------------
TZ = timezone(timedelta(hours=1))  # Africa/Lagos (fixed offset)
LATE_CUTOFF = dtime(9, 0, 0)       # 09:00 local
METHOD_USED = "facial_recognition"

# -------------------- GPIO/LCD/PiCam --------------------
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

# NOTE: keep address explicit to avoid surprises; if your LCD uses 0x3F, change it below.
lcd = CharLCD('PCF8574', 0x27)
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# -------------------- Utilities --------------------
def transition_to(state):
    print(f"[FSM] -> {state}")

def _today() -> date:
    return datetime.now(TZ).date()

def _now() -> datetime:
    return datetime.now(TZ)

def _attendance_path(d: date) -> str:
    return os.path.join(ATTEND_DIR, f"{d.isoformat()}.csv")

# 12-hour display format with AM/PM
TIME_FMT_12 = "%I:%M:%S %p"
# legacy support: old rows may be HH:MM:SS 24h
TIME_FMT_24 = "%H:%M:%S"

def _format_time_12(dt: datetime) -> str:
    """Return a 12-hour time string like '02:35:22 PM'."""
    return dt.strftime(TIME_FMT_12)

def _parse_clock_time(s: str):
    """Parse a time string that might be 12h with AM/PM or legacy 24h."""
    if not s:
        return None
    for fmt in (TIME_FMT_12, TIME_FMT_24):
        try:
            return datetime.strptime(s, fmt).time()
        except Exception:
            continue
    return None

DAILY_HEADERS = [
    "id", "employee_id", "full_name", "attendance_date",
    "clock_in", "clock_out", "total_hours", "total_minutes", "duration",
    "is_late", "is_absent", "method", "created_at", "updated_at"
]

def _ensure_daily_file(d: date):
    p = _attendance_path(d)
    if not os.path.exists(p):
        with open(p, "w", newline='', encoding="utf-8") as f:
            f.write(",".join(DAILY_HEADERS) + "\n")

def _read_daily(d: date) -> pd.DataFrame:
    p = _attendance_path(d)
    if not os.path.exists(p):
        return pd.DataFrame(columns=DAILY_HEADERS)
    return pd.read_csv(p, dtype=str).fillna("")

def _fmt_hms(total_seconds: float) -> str:
    try:
        secs = int(round(total_seconds))
        return str(timedelta(seconds=secs))  # e.g., 0:03:07
    except Exception:
        return ""

def _mk_para(text, styles):
    from reportlab.platypus import Paragraph
    s = "" if text is None else str(text)
    if "T" in s and ":" in s:
        parts = s.split("T", 1)
        s = f"{parts[0]}<br/>{parts[1]}"
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(s, styles["BodyText"])

def _sorted_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.reindex(columns=DAILY_HEADERS)
    df = df.copy()
    if list(df.columns) != DAILY_HEADERS:
        df = df.reindex(columns=DAILY_HEADERS, fill_value="")
    def _time_key(x):
        t = _parse_clock_time(x)
        return t if t is not None else dtime.max
    df["__cin__"] = df["clock_in"].apply(_time_key)
    df = df.sort_values(by=["attendance_date", "full_name", "__cin__"], ascending=[True, True, True])
    df = df.drop(columns=["__cin__"])
    return df

def _write_daily_pdf(d: date, df: pd.DataFrame):
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        print("[WARN] reportlab not installed. Skipping daily PDF export.")
        return

    os.makedirs(DAILY_PDF_DIR, exist_ok=True)
    final_path = os.path.join(DAILY_PDF_DIR, f"daily_{d.isoformat()}.pdf")
    tmp_path = final_path + ".tmp"

    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 10.5

    doc = SimpleDocTemplate(
        tmp_path,
        pagesize=landscape(A4),
        leftMargin=18, rightMargin=18, topMargin=24, bottomMargin=18,
        title=f"Attendance {d.isoformat()}"
    )

    weekday = d.strftime("%A")
    content = [Paragraph(f"{weekday} Attendance - {d.isoformat()}", styles["Title"]), Spacer(1, 8)]

    view = _sorted_daily(df).fillna("").astype(str)
    cols = [c for c in DAILY_PDF_COLUMNS if c in view.columns]
    if not cols:
        print("[WARN] DAILY_PDF_COLUMNS not found in dataframe; falling back to all columns.")
        cols = list(view.columns)

    header = [Paragraph(h, styles["BodyText"]) for h in cols]
    body = [[_mk_para(row[c], styles) for c in cols] for _, row in view.iterrows()]
    data = [header] + body

    total_width = 806.0  # landscape A4 (~842) minus margins
    weights = {
        "full_name": 3.2,
        "attendance_date": 1.2,
        "clock_in": 1.0,
        "clock_out": 1.0,
        "duration": 1.4,
        "is_late": 0.8,
        "is_absent": 0.9,
    }
    w_sum = sum(weights.get(c, 1.0) for c in cols)
    col_widths = [total_width * (weights.get(c, 1.0) / w_sum) for c in cols]

    tbl = Table(data, repeatRows=1, colWidths=col_widths, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))

    content.append(tbl)
    doc.build(content)
    os.replace(tmp_path, final_path)
    print(f"[EXPORT] Daily PDF -> {final_path}")

def _write_daily(d: date, df: pd.DataFrame):
    view = _sorted_daily(df)
    if WRITE_CSV:
        view.to_csv(_attendance_path(d), index=False)
    if WRITE_DAILY_PDF:
        _write_daily_pdf(d, view)

def _to_bool_str(x: bool) -> str:
    return "TRUE" if x else "FALSE"

def _round_hours(minutes: int) -> float:
    return round(minutes / 60.0, 2)

def _ensure_daily_artifacts(d: date):
    _ensure_daily_file(d)
    df = _read_daily(d)
    if WRITE_DAILY_PDF:
        _write_daily_pdf(d, df)

def _init_today_artifacts():
    try:
        _ensure_daily_artifacts(_today())
    except Exception as e:
        print(f"[INIT] Failed to ensure today's artifacts: {e}")

# -------------------- Local Attendance "Repository" --------------------
class LocalAttendanceManager:
    """
    CSV/PDF adapter mirroring DB semantics:
    - Single open check-in per employee per day
    - Duplicate protection
    - Late flag on check-in after 09:00
    - total_hours = (clock_out - clock_in) in hours (2 d.p.)
    - Weekly & Monthly exports (CSV + optional PDF)
    - Optional daily absentees (is_absent=TRUE)
    """
    def __init__(self):
        pass

    def log_attendance(self, employee_id: str, full_name: str, log_type: str):
        today = _today()
        now = _now()
        _ensure_daily_file(today)
        df = _read_daily(today)

        emp_rows = df[df["employee_id"] == str(employee_id)]

        # -------- HARD DAILY LOCK: one row per employee per day --------
        if log_type == "Check-In":
            if not emp_rows.empty:
                print(f"[LOCK] {full_name} already has a record for {today}. Blocking new Check-In.")
                return "locked"

            rid = f"{employee_id}-{today.isoformat()}-{len(df)+1}"
            is_late = _to_bool_str(now.time() > LATE_CUTOFF)

            new_row = {
                "id": rid,
                "employee_id": str(employee_id),
                "full_name": full_name,
                "attendance_date": today.isoformat(),
                "clock_in": _format_time_12(now),  # 12-hour time
                "clock_out": "",
                "total_hours": "0.00",
                "total_minutes": "0.00",
                "duration": "",
                "is_late": is_late,
                "is_absent": "FALSE",
                "method": METHOD_USED,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        elif log_type == "Check-Out":
            if emp_rows.empty:
                print(f"[WARN] {full_name} has no record to close for {today}.")
                return "duplicate"

            open_rows = emp_rows[(emp_rows["clock_in"] != "") & (emp_rows["clock_out"] == "")]
            if open_rows.empty:
                print(f"[LOCK] {full_name} already checked out (or no open check-in). Blocking.")
                return "locked"

            idx = open_rows.index[-1]
            t_in = _parse_clock_time(df.at[idx, "clock_in"])
            if t_in is None:
                print(f"[ERROR] Cannot parse clock_in '{df.at[idx, 'clock_in']}' for {full_name}.")
                return False

            cin = datetime.combine(today, t_in, TZ)
            cout = now

            total_seconds = max((cout - cin).total_seconds(), 0.0)
            hours = round(total_seconds / 3600.0, 2)
            minutes = round(total_seconds / 60.0, 2)
            dur = _fmt_hms(total_seconds)

            df.at[idx, "clock_out"] = _format_time_12(cout)  # 12-hour time
            df.at[idx, "total_hours"] = f"{hours:.2f}"
            df.at[idx, "total_minutes"] = f"{minutes:.2f}"
            df.at[idx, "duration"] = dur
            df.at[idx, "updated_at"] = now.isoformat()
        else:
            print("[ERROR] Unknown log type.")
            return False

        _write_daily(today, df)
        print(f"[LOGGED] {log_type} for {full_name} ({employee_id})")
        return True

    def mark_daily_absentees(self, all_employee_ids: list, target_date: date):
        if target_date.weekday() >= 5:
            print(f"[ABSENT] weekend {target_date}, skipped.")
            return
        _ensure_daily_file(target_date)
        df = _read_daily(target_date)
        present_ids = set(df[df["clock_in"] != ""]["employee_id"].tolist())
        missing = [eid for eid in all_employee_ids if str(eid) not in present_ids]
        now = _now()

        if not missing:
            print(f"[ABSENT] none to mark on {target_date}")
            return

        rows = []
        for eid in missing:
            rows.append({
                "id": f"{eid}-{target_date.isoformat()}-A",
                "employee_id": str(eid),
                "full_name": eid_to_name.get(str(eid), str(eid).replace("_", " ")),
                "attendance_date": target_date.isoformat(),
                "clock_in": "",
                "clock_out": "",
                "total_hours": "0.00",
                "total_minutes": "0.00",
                "duration": "",
                "is_late": "FALSE",
                "is_absent": "TRUE",
                "method": METHOD_USED,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            })
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        _write_daily(target_date, df)
        print(f"[ABSENT] Marked {len(rows)} employees absent for {target_date}")

    def export_weekly(self, any_day: date = None, to_csv=True, to_pdf=True):
        day = any_day or _today()
        iso_year, iso_week, _ = day.isocalendar()
        monday = day - timedelta(days=day.weekday())
        days = [monday + timedelta(days=i) for i in range(7)]
        frames = [_read_daily(d) for d in days]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DAILY_HEADERS)

        base = os.path.join(REPORTS_DIR, f"weekly_{iso_year}-{iso_week:02d}")
        if to_csv:
            df.to_csv(base + ".csv", index=False)
            print(f"[EXPORT] CSV -> {base}.csv")
        if to_pdf:
            _write_pdf(base + ".pdf", df, title=f"Weekly Report {iso_year}-W{iso_week:02d}")

    def export_monthly(self, year: int, month: int, to_csv=True, to_pdf=True):
        start = date(year, month, 1)
        end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        days = (end - start).days
        frames = [_read_daily(start + timedelta(days=i)) for i in range(days)]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DAILY_HEADERS)

        base = os.path.join(REPORTS_DIR, f"monthly_{year}-{month:02d}")
        if to_csv:
            df.to_csv(base + ".csv", index=False)
            print(f"[EXPORT] CSV -> {base}.csv")
        if to_pdf:
            _write_pdf(base + ".pdf", df, title=f"Monthly Report {year}-{month:02d}")

# ---- PDF writer (weekly/monthly) ----
def _write_pdf(path: str, df: pd.DataFrame, title: str):
    """
    Weekly/Monthly PDF export using the SAME layout as the daily PDF:
    - landscape A4
    - compact column subset (DAILY_PDF_COLUMNS)
    - wrapped cells and alternating row backgrounds
    """
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        print("[WARN] reportlab not installed. Skipping PDF export.")
        return

    # Prepare atomic write
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"

    # Styles
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading  = 10.5

    # Helper (same as daily) to wrap cell text + split ISO datetimes
    def _mk_para(text):
        from reportlab.platypus import Paragraph
        s = "" if text is None else str(text)
        if "T" in s and ":" in s:
            d, t = s.split("T", 1)
            s = f"{d}<br/>{t}"
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(s, styles["BodyText"])

    # Document
    doc = SimpleDocTemplate(
        tmp_path,
        pagesize=landscape(A4),
        leftMargin=18, rightMargin=18, topMargin=24, bottomMargin=18,
        title=title
    )

    # Title
    content = [Paragraph(title, styles["Title"]), Spacer(1, 8)]

    # Same ordering/cleanup as daily table
    view = _sorted_daily(df).fillna("").astype(str)

    cols = [c for c in DAILY_PDF_COLUMNS if c in view.columns]
    if not cols:
        cols = list(view.columns)  # fallback (shouldn't happen)

    header = [Paragraph(h, styles["BodyText"]) for h in cols]
    body   = [[_mk_para(row[c]) for c in cols] for _, row in view.iterrows()]
    data   = [header] + body

    # Same proportional widths as daily
    total_width = 806.0  # landscape A4 minus margins
    weights = {
        "full_name": 3.2,
        "attendance_date": 1.2,
        "clock_in": 1.0,
        "clock_out": 1.0,
        "duration": 1.4,
        "is_late": 0.8,
        "is_absent": 0.9,
    }
    w_sum = sum(weights.get(c, 1.0) for c in cols)
    col_widths = [total_width * (weights.get(c, 1.0) / w_sum) for c in cols]

    tbl = Table(data, repeatRows=1, colWidths=col_widths, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),0.25,colors.grey),
        ('FONTSIZE',(0,0),(-1,-1),9),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke, colors.white]),
    ]))

    content.append(tbl)
    doc.build(content)
    os.replace(tmp_path, path)
    print(f"[EXPORT] PDF -> {path}")

# -------------------- Employees & Encodings --------------------
id_to_encoding = {}   # str(employee_id) -> np.ndarray
eid_to_name = {}      # str(employee_id) -> "First Last"
EMPLOYEES_ROOT = os.path.join(BASE_DIR, "employees")

def build_or_load_encodings_from_folders():
    global known_face_encodings, known_face_names, known_face_ids, id_to_encoding, eid_to_name

    if not os.path.isdir(EMPLOYEES_ROOT):
        os.makedirs(EMPLOYEES_ROOT, exist_ok=True)
        print(f"[SYNC] Created employees root: {EMPLOYEES_ROOT} (add folders like Shehu_Yusuf/)")

    encodings, names, ids = [], [], []

    for folder in os.listdir(EMPLOYEES_ROOT):
        folder_path = os.path.join(EMPLOYEES_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue

        full_name = folder.replace("_", " ").strip()   # LCD/log-friendly
        employee_id = folder                            # stable ID (folder name)

        for file in os.listdir(folder_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder_path, file)
            try:
                image = face_recognition.load_image_file(img_path)
                locs = face_recognition.face_locations(image)
                encs = face_recognition.face_encodings(image, locs)
                if encs:
                    enc = encs[0]
                    encodings.append(enc)
                    names.append(full_name)
                    ids.append(employee_id)
                    id_to_encoding[employee_id] = enc
                    eid_to_name[employee_id] = full_name
                    print(f"[OK] Encoded {full_name} from {file}")
            except Exception as e:
                print(f"[ERROR] Failed on {img_path}: {e}")

    known_face_encodings = encodings
    known_face_names = names
    known_face_ids = ids

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"ids": ids, "names": names, "encodings": encodings}, f)

    print(f"[SYNC] Saved {len(ids)} encodings -> {ENCODINGS_FILE}")
    encoding_ready.set()

def build_or_load_encodings():
    return build_or_load_encodings_from_folders()

def load_encodings_from_cache_or_build():
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
                if all(k in data for k in ("ids", "names", "encodings")):
                    global known_face_ids, known_face_names, known_face_encodings
                    known_face_ids = data["ids"]
                    known_face_names = data["names"]
                    known_face_encodings = data["encodings"]
                    for i, eid in enumerate(known_face_ids):
                        eid_to_name[str(eid)] = known_face_names[i]
                    print(f"[SYNC] Loaded {len(known_face_ids)} encodings from cache.")
                    encoding_ready.set()
                    return
        except Exception as e:
            print(f"[WARN] Failed to read encodings cache: {e}")
    build_or_load_encodings()

# -------------------- Face Recognition + LCD UX --------------------
def recognize_face(timeout=15, headless=True):
    start = time.time()
    while time.time() - start < timeout:
        frame = picam2.capture_array()
        resized = cv2.resize(frame, (0, 0), fx=1 / cv_scaler, fy=1 / cv_scaler)
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
            if len(known_face_encodings) == 0:
                continue
            dists = face_recognition.face_distance(known_face_encodings, enc)
            j = int(np.argmin(dists))
            if dists[j] <= TOLERANCE:
                if not headless:
                    cv2.destroyAllWindows()
                return known_face_ids[j], known_face_names[j], round((1 - dists[j]) * 100, 2)

    if not headless:
        cv2.destroyAllWindows()
    return None, None, None

def display(lines, delay=0.3):
    with lcd_lock:
        lcd.clear()
        for line in lines:
            if len(line) <= 16:
                lcd.write_string(line.ljust(16))
                lcd.crlf()
            else:
                for i in range(len(line) - 15):
                    lcd.clear()
                    lcd.write_string(line[i:i + 16])
                    lcd.crlf()
                    time.sleep(delay)
                time.sleep(1)

def get_greeting_lines():
    hour = _now().hour
    if 5 <= hour < 12:
        return ["GOOD MORNING!", "PRESS TO LOG"]
    elif 12 <= hour < 17:
        return ["GOOD AFTERNOON!", "PRESS TO LOG"]
    elif 17 <= hour < 21:
        return ["GOOD EVENING!", "PRESS TO LOG"]
    else:
        return ["WELCOME BACK!", "PRESS TO LOG"]

def hourly_greeting_updater():
    last_hour = -1
    while True:
        current_hour = _now().hour
        if current_hour != last_hour:
            display(get_greeting_lines())
            last_hour = current_hour
        time.sleep(30)

# -------------------- Button Handlers --------------------
def handle_attendance(mgr: 'LocalAttendanceManager', action: str, headless=True):
    print(f"[DEBUG] Button triggered action: {action}")
    display(["LOOK AT THE CAMERA"])
    eid, full_name, confidence = recognize_face(headless=headless)
    if full_name:
        display([f"DETECTED: {full_name[:16].upper()}", f"MATCH: {confidence}%"])
        time.sleep(1.2)
        result = mgr.log_attendance(str(eid), full_name, action)
        if result in ("duplicate", "locked"):
            display(["ALREADY LOGGED", "FOR TODAY"])
            time.sleep(2)
            display(get_greeting_lines())
            transition_to("IDLE")
            return
        elif result:
            GPIO.output(GREEN_LED_PIN if action == "Check-In" else RED_LED_PIN, GPIO.HIGH)
            display([f"{full_name.split()[0][:12].upper()} {action.upper()}",
                     "SUCCESS! PDF SAVED"])
            time.sleep(2.2)
            GPIO.output(GREEN_LED_PIN if action == "Check-In" else RED_LED_PIN, GPIO.LOW)
        else:
            display(["LOG FAILED", "TRY AGAIN"])
    else:
        display(["FACE NOT", "RECOGNIZED"])
    time.sleep(1.5)
    display(get_greeting_lines())
    transition_to("IDLE")

# -------------------- Backfill helpers --------------------
def _weekly_expected_paths(any_day: date):
    iso_year, iso_week, _ = any_day.isocalendar()
    base = os.path.join(REPORTS_DIR, f"weekly_{iso_year}-{iso_week:02d}")
    return base + ".pdf", base + ".csv"

def _monthly_expected_paths(year: int, month: int):
    base = os.path.join(REPORTS_DIR, f"monthly_{year}-{month:02d}")
    return base + ".pdf", base + ".csv"

def _backfill_weekly_if_missing(mgr: 'LocalAttendanceManager'):
    ref = _today() - timedelta(days=3)  # last Friday -> previous week
    pdf_path, csv_path = _weekly_expected_paths(ref)
    if not (os.path.exists(pdf_path) or os.path.exists(csv_path)):
        print("[BACKFILL] Weekly report missing; generating now.")
        mgr.export_weekly(any_day=ref, to_csv=True, to_pdf=True)

def _backfill_monthly_if_missing(mgr: 'LocalAttendanceManager'):
    today = _today()
    prev_month_last = (today.replace(day=1) - timedelta(days=1))
    y, m = prev_month_last.year, prev_month_last.month
    pdf_path, csv_path = _monthly_expected_paths(y, m)
    if not (os.path.exists(pdf_path) or os.path.exists(csv_path)):
        print("[BACKFILL] Monthly report missing; generating now.")
        mgr.export_monthly(y, m, to_csv=True, to_pdf=True)

# -------------------- Minimal freeze-proofing --------------------
def _clock_sane(min_year: int = 2020) -> bool:
    """True if system clock looks reasonable (avoids APS catch-up storms on 1970 time)."""
    try:
        return _now().year >= min_year
    except Exception:
        return False

# -------------------- Main --------------------
def main():
    mgr = LocalAttendanceManager()

    # Encodings: load cache or build from local employee folders
    load_encodings_from_cache_or_build()

    # Make scheduler timezone-aware and resilient to misfires.
    scheduler = BackgroundScheduler(
        timezone=TZ,
        job_defaults={
            "coalesce": True,
            "misfire_grace_time": 3600  # 1 hour; prevents massive catch-up on wrong clocks
        }
    )

    # Only register and start scheduled jobs if the clock is sane.
    if _clock_sane():
        # Periodic refresh of encodings
        scheduler.add_job(build_or_load_encodings, 'interval', minutes=10)

        # On-time schedules
        scheduler.add_job(lambda: mgr.export_weekly(to_csv=True, to_pdf=True),
                          'cron', day_of_week='fri', hour=22, minute=0)
        scheduler.add_job(lambda: mgr.export_monthly(_today().year, _today().month, to_csv=True, to_pdf=True),
                          'cron', day='last', hour=22, minute=0)

        # Catch-up schedules if Pi was off
        scheduler.add_job(lambda: mgr.export_weekly(any_day=_today() - timedelta(days=3), to_csv=True, to_pdf=True),
                          'cron', day_of_week='mon', hour=8, minute=0)
        scheduler.add_job(
            lambda: (lambda pm=(_today().replace(day=1) - timedelta(days=1)):
                     mgr.export_monthly(pm.year, pm.month, to_csv=True, to_pdf=True))(),
            'cron', day=1, hour=8, minute=0)

        # Daily absent marking & day file bootstrap
        def _mark_absents():
            if not os.path.isdir(EMPLOYEES_ROOT):
                return
            all_ids = [d for d in os.listdir(EMPLOYEES_ROOT)
                       if os.path.isdir(os.path.join(EMPLOYEES_ROOT, d))]
            mgr.mark_daily_absentees(all_ids, _today())
        scheduler.add_job(_mark_absents, 'cron', hour=23, minute=59)

        # Bootstrap & daily artifact ensure
        _init_today_artifacts()
        scheduler.add_job(lambda: _ensure_daily_artifacts(_today()), 'cron', hour=0, minute=1)

        try:
            scheduler.start()
        except Exception as e:
            print(f"[SCHED] start failed: {e}. Continuing without scheduler.")
    else:
        print("[SCHED] System clock not sane; running attendance loop without scheduler jobs.")
        _init_today_artifacts()

    # LCD greeting thread after encodings are ready
    threading.Thread(target=lambda: (encoding_ready.wait(), hourly_greeting_updater()), daemon=True).start()

    try:
        display(get_greeting_lines())
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
