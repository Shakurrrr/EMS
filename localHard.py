# -*- coding: utf-8 -*-

import os
import time
import pickle
import threading
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

# Tuned for stricter but still responsive recognition
TOLERANCE = 0.41               # lower => stricter (0.35Â–0.45 typical)
SECOND_BEST_MARGIN = 0.05      # best must beat 2nd-best *other person* by this much
MATCH_STREAK = 2               # frames needed to lock the same person
MAX_FRAMES = 20                # cap per recognition attempt
SINGLE_FACE_ONLY = True        # require exactly one face in frame

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

# NEW: daily auto-close cutoff (office close)
AUTOCLOSE_CUTOFF = dtime(17, 0, 0)  # 05:00 PM

# -------------------- Freeze-proofing & Repair Controls --------------------
AUTO_CORRECT_BAD_TIME = True   # enable auto-fix when clock becomes sane
REPAIR_POLL_SEC = 10           # how often to check for sane clock and trigger repair

# For deferred scheduler start
_GLOBAL_SCHEDULER = None
_GLOBAL_MGR = None
_SCHEDULER_STARTED = False

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

# NOTE: keep address explicit; if your LCD uses 0x3F, change it below.
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
    return dt.strftime(TIME_FMT_12)

def _parse_clock_time(s: str):
    if not s:
        return None
    for fmt in (TIME_FMT_12, TIME_FMT_24):
        try:
            return datetime.strptime(s, fmt).time()
        except Exception:
            continue
    return None

# System clock sanity gate (prevents APScheduler storms on 1970 time)
def _clock_sane(min_year: int = 2020) -> bool:
    try:
        return _now().year >= min_year
    except Exception:
        return False

def _current_boot_id() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id", "r") as f:
            return f.read().strip()
    except Exception:
        return ""

# Include repair metadata at the end to preserve CSV compatibility order
DAILY_HEADERS = [
    "id", "employee_id", "full_name", "attendance_date",
    "clock_in", "clock_out", "total_hours", "total_minutes", "duration",
    "is_late", "is_absent", "method", "created_at", "updated_at",
    "time_quality", "mono_in_ns", "mono_out_ns", "boot_id", "time_corrected"
]

def _ensure_daily_file(d: date):
    p = _attendance_path(d)
    if not os.path.exists(p):
        with open(p, "w", newline='', encoding="utf-8") as f:
            f.write(",".join(DAILY_HEADERS) + "\n")

def _read_daily(d: date) -> pd.DataFrame:
    """
    Safely read a daily CSV.  Re-initializes headers if the file is empty,
    unreadable, or missing columns.
    """
    p = _attendance_path(d)
    if not os.path.exists(p):
        return pd.DataFrame(columns=DAILY_HEADERS)

    try:
        if os.path.getsize(p) == 0:
            print(f"[WARN] {p} is empty; reinitializing headers.")
            _ensure_daily_file(d)
            return pd.DataFrame(columns=DAILY_HEADERS)
    except Exception as e:
        print(f"[WARN] Could not stat {p}: {e}")

    try:
        df = pd.read_csv(p, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        print(f"[WARN] {p} has no columns; reinitializing headers.")
        _ensure_daily_file(d)
        return pd.DataFrame(columns=DAILY_HEADERS)
    except Exception as e:
        print(f"[ERROR] Failed to read {p}: {e}; reinitializing headers.")
        _ensure_daily_file(d)
        return pd.DataFrame(columns=DAILY_HEADERS)

    if list(df.columns) != DAILY_HEADERS:
        df = df.reindex(columns=DAILY_HEADERS, fill_value="")
    return df


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

    from reportlab.platypus import Table, TableStyle
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

# -------------------- Time Repair Logic --------------------
def _monotonic_to_wall(mono_ns_str: str):
    if not mono_ns_str:
        return None
    try:
        mn = time.monotonic_ns()
        wn = _now()
        me = int(mono_ns_str)
        delta_sec = (mn - me) / 1_000_000_000.0
        return wn - timedelta(seconds=delta_sec)
    except Exception:
        return None

def _recalc_is_late(clock_in_dt: datetime) -> str:
    try:
        return "TRUE" if clock_in_dt.time() > LATE_CUTOFF else "FALSE"
    except Exception:
        return "FALSE"

def _autocorrect_bad_times():
    if not _clock_sane():
        return

    boot_id = _current_boot_id()
    for fname in os.listdir(ATTEND_DIR):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(ATTEND_DIR, fname)
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
        except Exception as e:
            print(f"[REPAIR] Cannot read {path}: {e}")
            continue

        if "time_quality" not in df.columns:
            continue

        changed_rows = []
        for idx, row in df.iterrows():
            if row.get("time_quality", "") != "UNSANE":
                continue
            if row.get("boot_id", "") != boot_id:
                continue

            cin_wall = _monotonic_to_wall(row.get("mono_in_ns", ""))
            cout_wall = _monotonic_to_wall(row.get("mono_out_ns", ""))
            if cin_wall is None:
                continue

            df.at[idx, "clock_in"] = _format_time_12(cin_wall)
            if cout_wall is not None:
                df.at[idx, "clock_out"] = _format_time_12(cout_wall)
                total_seconds = max((cout_wall - cin_wall).total_seconds(), 0.0)
                df.at[idx, "total_hours"] = f"{(total_seconds/3600.0):.2f}"
                df.at[idx, "total_minutes"] = f"{(total_seconds/60.0):.2f}"
                df.at[idx, "duration"] = _fmt_hms(total_seconds)

            df.at[idx, "attendance_date"] = cin_wall.date().isoformat()
            df.at[idx, "created_at"] = cin_wall.isoformat()
            df.at[idx, "updated_at"] = (_now()).isoformat()
            df.at[idx, "is_late"] = _recalc_is_late(cin_wall)
            df.at[idx, "time_quality"] = "SANE"
            df.at[idx, "time_corrected"] = "TRUE"
            changed_rows.append(idx)

        if not changed_rows:
            continue

        # Repartition rows per attendance_date and write out fresh CSVs
        by_date = {}
        for _, r in df.iterrows():
            target_date = r.get("attendance_date", "")
            by_date.setdefault(target_date, []).append(r)

        for target_date, rows in by_date.items():
            if not target_date:
                continue
            target_path = os.path.join(ATTEND_DIR, f"{target_date}.csv")
            out_df = pd.DataFrame(rows)
            if list(out_df.columns) != DAILY_HEADERS:
                out_df = out_df.reindex(columns=DAILY_HEADERS, fill_value="")
            out_df.to_csv(target_path, index=False)

        try:
            os.remove(path)
        except Exception:
            pass

        print(f"[REPAIR] Corrected {len(changed_rows)} rows using monotonic mapping in {fname}")

def _clock_repair_monitor():
    global _GLOBAL_SCHEDULER, _GLOBAL_MGR, _SCHEDULER_STARTED
    was_sane = _clock_sane()
    while True:
        now_sane = _clock_sane()
        if not was_sane and now_sane:
            print("[REPAIR] Clock transitioned to sane.")
            if AUTO_CORRECT_BAD_TIME:
                _autocorrect_bad_times()
            if _GLOBAL_SCHEDULER is not None and not _SCHEDULER_STARTED:
                try:
                    _register_scheduler_jobs(_GLOBAL_SCHEDULER, _GLOBAL_MGR)
                    _GLOBAL_SCHEDULER.start()
                    _SCHEDULER_STARTED = True
                    print("[SCHED] Started scheduler after clock became sane.]")
                except Exception as e:
                    print(f"[SCHED] deferred start failed: {e}")
        was_sane = now_sane
        time.sleep(REPAIR_POLL_SEC)

# -------------------- Weekly Report Helpers --------------------
def _week_bounds(any_day: date):
    monday = any_day - timedelta(days=any_day.weekday())
    days = [monday + timedelta(days=i) for i in range(7)]
    return monday.isocalendar(), days

def _build_weekly_frames(any_day: date) -> pd.DataFrame:
    _, days = _week_bounds(any_day)
    frames = [_read_daily(d) for d in days]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DAILY_HEADERS)

def _weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["full_name", "week_total_hours"])
    tmp = df.copy()
    tmp["full_name"] = tmp["full_name"].fillna("").astype(str)
    tmp["total_hours"] = pd.to_numeric(tmp["total_hours"], errors="coerce").fillna(0.0)
    summary = tmp.groupby("full_name", as_index=False)["total_hours"].sum()
    summary = summary.rename(columns={"total_hours": "week_total_hours"})
    summary["week_total_hours"] = summary["week_total_hours"].round(2)
    return summary.sort_values("full_name").reset_index(drop=True)

def _write_weekly_pdf(path: str, logs_df: pd.DataFrame, summary_df: pd.DataFrame, title: str):
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        print("[WARN] reportlab not installed. Skipping weekly PDF export.")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"

    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading  = 12

    doc = SimpleDocTemplate(
        tmp_path,
        pagesize=landscape(A4),
        leftMargin=18, rightMargin=18, topMargin=24, bottomMargin=18,
        title=title
    )

    content = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

    # Summary table
    sum_header = [Paragraph("Full Name", styles["BodyText"]), Paragraph("Week Total (h)", styles["BodyText"])]
    sum_body = [[Paragraph(str(r["full_name"]), styles["BodyText"]),
                 Paragraph(f"{float(r['week_total_hours']):.2f}", styles["BodyText"])]
                for _, r in summary_df.iterrows()]
    from reportlab.platypus import Table, TableStyle
    sum_tbl = Table([sum_header] + sum_body, repeatRows=1, hAlign="LEFT")
    sum_tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),0.25,colors.grey),
        ('FONTSIZE',(0,0),(-1,-1),10),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke, colors.white]),
    ]))
    content += [Paragraph("Summary (Per Employee)", styles["Heading2"]), sum_tbl, Spacer(1, 16)]

    # Detailed logs
    view = _sorted_daily(logs_df).fillna("").astype(str)
    det_cols = ["full_name", "attendance_date", "clock_in", "clock_out", "duration", "is_late", "is_absent"]
    det_header = [Paragraph(c.replace("_"," "), styles["BodyText"]) for c in det_cols]
    def _mk(text):
        from reportlab.platypus import Paragraph
        return Paragraph(str(text), styles["BodyText"])
    det_body = [[_mk(row[c]) for c in det_cols] for _, row in view.iterrows()]
    det_tbl = Table([det_header] + det_body, repeatRows=1, hAlign="LEFT")
    det_tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),0.25,colors.grey),
        ('FONTSIZE',(0,0),(-1,-1),9),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke, colors.white]),
    ]))
    content += [Paragraph("Detailed Logs", styles["Heading2"]), det_tbl]

    doc.build(content)
    os.replace(tmp_path, path)
    print(f"[EXPORT] Weekly PDF -> {path}")

# ---- PDF writer (weekly/monthly generic, legacy) ----
def _write_pdf(path: str, df: pd.DataFrame, title: str):
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        print("[WARN] reportlab not installed. Skipping PDF export.")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"

    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading  = 10.5

    def _mk_para(text):
        from reportlab.platypus import Paragraph
        s = "" if text is None else str(text)
        if "T" in s and ":" in s:
            d, t = s.split("T", 1)
            s = f"{d}<br/>{t}"
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(s, styles["BodyText"])

    doc = SimpleDocTemplate(
        tmp_path,
        pagesize=landscape(A4),
        leftMargin=18, rightMargin=18, topMargin=24, bottomMargin=18,
        title=title
    )

    content = [Paragraph(title, styles["Title"]), Spacer(1, 8)]

    view = _sorted_daily(df).fillna("").astype(str)
    cols = [c for c in DAILY_PDF_COLUMNS if c in view.columns]
    if not cols:
        cols = list(view.columns)

    header = [Paragraph(h, styles["BodyText"]) for h in cols]
    body   = [[_mk_para(row[c]) for c in cols] for _, row in view.iterrows()]
    data   = [header] + body

    total_width = 806.0
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

    from reportlab.platypus import Table, TableStyle
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

# -------------------- MONTHLY SUMMARY (NEW) --------------------
def _monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["full_name", "month_total_hours"])
    tmp = df.copy()
    tmp["full_name"] = tmp["full_name"].fillna("").astype(str)
    tmp["total_hours"] = pd.to_numeric(tmp["total_hours"], errors="coerce").fillna(0.0)
    summary = tmp.groupby("full_name", as_index=False)["total_hours"].sum()
    summary = summary.rename(columns={"total_hours": "month_total_hours"})
    summary["month_total_hours"] = summary["month_total_hours"].round(2)
    return summary.sort_values("full_name").reset_index(drop=True)

def _write_monthly_summary_pdf(path: str, summary_df: pd.DataFrame, title: str):
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        print("[WARN] reportlab not installed. Skipping monthly PDF export.")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"

    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 12

    doc = SimpleDocTemplate(
        tmp_path,
        pagesize=landscape(A4),
        leftMargin=18, rightMargin=18, topMargin=24, bottomMargin=18,
        title=title
    )

    content = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

    header = [
        Paragraph("Full Name", styles["BodyText"]),
        Paragraph("Month Total (h)", styles["BodyText"]),
    ]
    body = [
        [Paragraph(str(r["full_name"]), styles["BodyText"]),
         Paragraph(f"{float(r['month_total_hours']):.2f}", styles["BodyText"])]
        for _, r in summary_df.iterrows()
    ]

    from reportlab.platypus import Table, TableStyle
    tbl = Table([header] + body, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))

    content.append(tbl)
    doc.build(content)
    os.replace(tmp_path, path)
    print(f"[EXPORT] Monthly Summary PDF -> {path}")

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

        full_name = folder.replace("_", " ").strip()
        employee_id = folder

        for file in os.listdir(folder_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder_path, file)
            try:
                image = face_recognition.load_image_file(img_path)
                locs = face_recognition.face_locations(image)
                encs = face_recognition.face_encodings(image, locs, num_jitters=2)
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
def recognize_face(timeout=10, headless=True):
    """
    Returns (employee_id, full_name, confidence_percent)
    after requiring a small streak of consistent matches across frames.
    """
    start = time.time()
    votes = {}
    best_conf = {}

    frames_seen = 0
    while time.time() - start < timeout and frames_seen < MAX_FRAMES:
        frame = picam2.capture_array()
        resized = cv2.resize(frame, (0, 0), fx=1 / cv_scaler, fy=1 / cv_scaler)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model='hog')

        if SINGLE_FACE_ONLY and len(locs) != 1:
            frames_seen += 1
            if not headless:
                for (t, r, b, l) in locs:
                    cv2.rectangle(resized, (l, t), (r, b), (0, 255, 0), 2)
                cv2.imshow("Camera Preview", resized)
                cv2.waitKey(1)
            continue

        encs = face_recognition.face_encodings(rgb, locs)
        if not headless:
            for (t, r, b, l) in locs:
                cv2.rectangle(resized, (l, t), (r, b), (0, 255, 0), 2)
            cv2.imshow("Camera Preview", resized)
            cv2.waitKey(1)

        frames_seen += 1
        if not encs or len(known_face_encodings) == 0:
            continue

        for enc in encs:
            dists = face_recognition.face_distance(known_face_encodings, enc)
            if dists is None or len(dists) == 0:
                continue

            # Per-ID minimum distance
            per_id_min = {}
            for idx, dist in enumerate(dists):
                eid_k = known_face_ids[idx]
                cur = per_id_min.get(eid_k, 1e9)
                if dist < cur:
                    per_id_min[eid_k] = float(dist)

            best_eid, best = min(per_id_min.items(), key=lambda kv: kv[1])
            others = [v for k, v in per_id_min.items() if k != best_eid]
            second = min(others) if others else None

            if best > TOLERANCE:
                continue
            if second is not None and (second - best) < SECOND_BEST_MARGIN:
                continue

            votes[best_eid] = votes.get(best_eid, 0) + 1
            conf = 1.0 - best
            if best_eid not in best_conf or conf > best_conf[best_eid]:
                best_conf[best_eid] = conf

            if votes[best_eid] >= MATCH_STREAK:
                name = eid_to_name.get(str(best_eid), str(best_eid).replace("_", " "))
                if not headless:
                    cv2.destroyAllWindows()
                return best_eid, name, round(best_conf[best_eid] * 100.0, 2)

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

# -------------------- Scheduler job registration --------------------
def _register_scheduler_jobs(scheduler, mgr):
    # periodic encodings refresh
    scheduler.add_job(build_or_load_encodings, 'interval', minutes=120)

    # on-time schedules (Friday 22:00)
    scheduler.add_job(lambda: mgr.export_weekly(to_csv=True, to_pdf=True),
                      'cron', day_of_week='fri', hour=22, minute=0)
    scheduler.add_job(lambda: mgr.export_monthly(_today().year, _today().month, to_csv=True, to_pdf=True),
                      'cron', day='last', hour=22, minute=0)

    # catch-up schedule (Monday morning regen of previous week in case Pi was off)
    scheduler.add_job(lambda: mgr.export_weekly(any_day=_today() - timedelta(days=3), to_csv=True, to_pdf=True),
                      'cron', day_of_week='mon', hour=8, minute=0)

    # Daily absent marking & day file bootstrap
    def _mark_absents():
        if not os.path.isdir(EMPLOYEES_ROOT):
            return
        all_ids = [d for d in os.listdir(EMPLOYEES_ROOT)
                   if os.path.isdir(os.path.join(EMPLOYEES_ROOT, d))]
        if all_ids:
            mgr.mark_daily_absentees(all_ids, _today())
    scheduler.add_job(_mark_absents, 'cron', hour=23, minute=59)

    # daily artifact ensure
    scheduler.add_job(lambda: _ensure_daily_artifacts(_today()), 'cron', hour=0, minute=1)

    # TIME-AGNOSTIC AUTOCLOSE: runs whenever device is ON (idempotent)
    scheduler.add_job(lambda: mgr.autoclose_previous_day(cutoff=AUTOCLOSE_CUTOFF),
                      'interval', minutes=300)

# -------------------- Weekly auto-refresh after logs --------------------
_LAST_WEEKLY_REFRESH_TS = 0
WEEKLY_REFRESH_COOLDOWN_SEC = 300  # 5 minutes

def _maybe_refresh_current_week(mgr):
    global _LAST_WEEKLY_REFRESH_TS
    now_ts = time.time()
    if now_ts - _LAST_WEEKLY_REFRESH_TS >= WEEKLY_REFRESH_COOLDOWN_SEC:
        try:
            mgr.export_weekly(any_day=_today(), to_csv=True, to_pdf=True)
            _LAST_WEEKLY_REFRESH_TS = now_ts
            print("[WEEKLY] Auto-refreshed current week.")
        except Exception as e:
            print(f"[WEEKLY] Auto-refresh failed: {e}")

# -------------------- Local Attendance Repository --------------------
class LocalAttendanceManager:
    """
    CSV/PDF adapter mirroring DB semantics:
    - Single open check-in per employee per day
    - Duplicate protection
    - Late flag on check-in after 09:00
    - total_hours = (clock_out - clock_in) in hours (2 d.p.)
    - Weekly & Monthly exports (CSV + PDF)
    - Optional daily absentees (is_absent=TRUE)
    - AUTOCLOSE: backfill 06:00:00 PM for prior day if no clock_out
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

            quality = "SANE" if _clock_sane() else "UNSANE"
            mono_now = str(time.monotonic_ns())

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
                "time_quality": quality,
                "mono_in_ns": mono_now,
                "mono_out_ns": "",
                "boot_id": _current_boot_id(),
                "time_corrected": "FALSE",
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

            # capture monotonic and propagate quality
            df.at[idx, "mono_out_ns"] = str(time.monotonic_ns())
            prev_q = df.at[idx, "time_quality"] if "time_quality" in df.columns else "SANE"
            df.at[idx, "time_quality"] = "SANE" if (prev_q == "SANE" and _clock_sane()) else "UNSANE"
        else:
            print("[ERROR] Unknown log type.")
            return False

        _write_daily(today, df)
        _maybe_refresh_current_week(self)
        print(f"[LOGGED] {log_type} for {full_name} ({employee_id})")
        return True

    # -------------------- AUTOCLOSE CORE --------------------
    def autoclose_for_day(self, target_date: date, cutoff_time: dtime = AUTOCLOSE_CUTOFF) -> int:
        """
        For the given target_date, find rows with clock_in set and clock_out empty,
        set clock_out to cutoff_time (default 06:00:00 PM), recompute totals,
        and regenerate that day's CSV + PDF. Returns number of rows updated.
        """
        _ensure_daily_file(target_date)
        df = _read_daily(target_date)
        if df.empty:
            return 0

        updates = 0
        now_iso = _now().isoformat()

        for idx, row in df.iterrows():
            # Skip absences and rows without an open check-in
            if row.get("is_absent", "").upper() == "TRUE":
                continue
            cin_str = row.get("clock_in", "")
            cout_str = row.get("clock_out", "")
            if not cin_str or cout_str:
                continue

            tin = _parse_clock_time(cin_str)
            if tin is None:
                continue

            cin_dt = datetime.combine(target_date, tin, TZ)
            cutoff_dt = datetime.combine(target_date, cutoff_time, TZ)
            cout_dt = cutoff_dt if cutoff_dt >= cin_dt else cin_dt  # never negative duration

            total_seconds = max((cout_dt - cin_dt).total_seconds(), 0.0)
            hours = round(total_seconds / 3600.0, 2)
            minutes = round(total_seconds / 60.0, 2)
            dur = _fmt_hms(total_seconds)

            df.at[idx, "clock_out"] = _format_time_12(cout_dt)  # e.g., "06:00:00 PM"
            df.at[idx, "total_hours"] = f"{hours:.2f}"
            df.at[idx, "total_minutes"] = f"{minutes:.2f}"
            df.at[idx, "duration"] = dur
            df.at[idx, "updated_at"] = now_iso

            prev_q = df.at[idx, "time_quality"] if "time_quality" in df.columns else "SANE"
            df.at[idx, "time_quality"] = "SANE" if prev_q == "SANE" else prev_q
            # mono_out_ns intentionally left blank for autoclose

            updates += 1

        if updates > 0:
            _write_daily(target_date, df)  # regenerates CSV + PDF
            try:
                self.export_weekly(any_day=target_date, to_csv=True, to_pdf=True)
                print(f"[AUTOCLOSE] Weekly refreshed due to changes on {target_date}")
            except Exception as e:
                print(f"[AUTOCLOSE] Weekly refresh failed for {target_date}: {e}")

        print(f"[AUTOCLOSE] Closed {updates} open row(s) for {target_date}")
        return updates

    def autoclose_previous_day(self, cutoff: dtime = AUTOCLOSE_CUTOFF) -> int:
        """Convenience wrapper to auto-close the previous calendar day."""
        target_date = _today() - timedelta(days=1)
        return self.autoclose_for_day(target_date, cutoff_time=cutoff)

    # -------------------- /AUTOCLOSE --------------------

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
                "time_quality": "SANE" if _clock_sane() else "UNSANE",
                "mono_in_ns": "",
                "mono_out_ns": "",
                "boot_id": _current_boot_id(),
                "time_corrected": "FALSE",
            })
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        _write_daily(target_date, df)
        print(f"[ABSENT] Marked {len(rows)} employees absent for {target_date}")

    def export_weekly(self, any_day: date = None, to_csv=True, to_pdf=True):
        day = any_day or _today()
        iso_year, iso_week, _ = day.isocalendar()
        logs_df = _build_weekly_frames(day)
        summary_df = _weekly_summary(logs_df)

        base = os.path.join(REPORTS_DIR, f"weekly_{iso_year}-{iso_week:02d}")
        if to_csv:
            logs_df.to_csv(base + ".csv", index=False)
            summary_df.to_csv(base + "_summary.csv", index=False)
            print(f"[EXPORT] CSV -> {base}.csv ; {base}_summary.csv")
        if to_pdf:
            _write_weekly_pdf(base + ".pdf", logs_df, summary_df, title=f"Weekly Report {iso_year}-W{iso_week:02d}")

    def export_monthly(self, year: int, month: int, to_csv=True, to_pdf=True):
        start = date(year, month, 1)
        end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        days = (end - start).days
        frames = [_read_daily(start + timedelta(days=i)) for i in range(days)]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DAILY_HEADERS)

        summary_df = _monthly_summary(df)

        base = os.path.join(REPORTS_DIR, f"monthly_{year}-{month:02d}")
        if to_csv:
            df.to_csv(base + ".csv", index=False)
            summary_df.to_csv(base + "_summary.csv", index=False)
            print(f"[EXPORT] CSV -> {base}.csv ; {base}_summary.csv")
        if to_pdf:
            _write_monthly_summary_pdf(base + ".pdf", summary_df, title=f"Monthly Summary {year}-{month:02d}")

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

def recalc_and_regen_day(target_date: date):
    """Recompute totals from CSV for a given day and rebuild that day's PDF.
       Also refresh the week that contains that day."""
    _ensure_daily_file(target_date)
    df = _read_daily(target_date)

    for idx, row in df.iterrows():
        tin  = _parse_clock_time(row.get("clock_in", ""))
        tout = _parse_clock_time(row.get("clock_out", ""))
        if tin and tout:
            cin  = datetime.combine(target_date, tin,  TZ)
            cout = datetime.combine(target_date, tout, TZ)
            total_seconds = max((cout - cin).total_seconds(), 0.0)
            df.at[idx, "total_hours"]   = f"{(total_seconds/3600.0):.2f}"
            df.at[idx, "total_minutes"] = f"{(total_seconds/60.0):.2f}"
            df.at[idx, "duration"]      = _fmt_hms(total_seconds)
            df.at[idx, "is_late"]       = _to_bool_str(tin > LATE_CUTOFF)
            df.at[idx, "updated_at"]    = _now().isoformat()

    _write_daily(target_date, df)  # writes CSV + PDF

    try:
        mgr = _GLOBAL_MGR or LocalAttendanceManager()
        mgr.export_weekly(any_day=target_date, to_csv=True, to_pdf=True)
        print(f"[WEEKLY] Regen because {target_date} changed.")
    except Exception as e:
        print(f"[WEEKLY] Regen failed for {target_date}: {e}")

def _regen_recent_days(n=3):
    for i in range(1, n+1):
        try:
            d = _today() - timedelta(days=i)
            p_csv = _attendance_path(d)
            p_pdf = os.path.join(REPORTS_DIR, f"daily_{d.isoformat()}.pdf")
            if os.path.exists(p_csv) and (not os.path.exists(p_pdf) or os.path.getmtime(p_csv) > os.path.getmtime(p_pdf)):
                print(f"[REGEN] CSV newer than PDF for {d}, rebuilding...")
                recalc_and_regen_day(d)
        except Exception as e:
            print(f"[REGEN] Failed for {d}: {e}")

# -------------------- Main --------------------
def main():
    global _GLOBAL_SCHEDULER, _GLOBAL_MGR, _SCHEDULER_STARTED

    mgr = LocalAttendanceManager()
    _GLOBAL_MGR = mgr

    # Encodings: load cache or build from local employee folders
    load_encodings_from_cache_or_build()

    # Make scheduler timezone-aware and resilient to misfires.
    scheduler = BackgroundScheduler(
        timezone=TZ,
        job_defaults={
            "coalesce": True,
            "misfire_grace_time": 3600  # prevents massive catch-up on wrong clocks
        }
    )
    _GLOBAL_SCHEDULER = scheduler

    # Only register and start scheduled jobs if the clock is sane.
    if _clock_sane():
        _register_scheduler_jobs(scheduler, mgr)
        try:
            scheduler.start()
            _SCHEDULER_STARTED = True
        except Exception as e:
            print(f"[SCHED] start failed: {e}. Continuing without scheduler.")
        _init_today_artifacts()
        _regen_recent_days(n=3)
        # BOOT-TIME CATCH-UP: sweep last 3 days for missed autoclose
        try:
            mgr.autoclose_for_day(_today() - timedelta(days=1), cutoff_time=AUTOCLOSE_CUTOFF)
            mgr.autoclose_for_day(_today() - timedelta(days=2), cutoff_time=AUTOCLOSE_CUTOFF)
            mgr.autoclose_for_day(_today() - timedelta(days=3), cutoff_time=AUTOCLOSE_CUTOFF)
        except Exception as e:
            print(f"[BOOT] Autoclose catch-up failed: {e}")
    else:
        print("[SCHED] System clock not sane; running attendance loop without scheduler jobs.")
        _init_today_artifacts()
        _regen_recent_days(n=3)
        # Even if scheduler is off, perform a one-shot catch-up at boot
        try:
            mgr.autoclose_for_day(_today() - timedelta(days=1), cutoff_time=AUTOCLOSE_CUTOFF)
            mgr.autoclose_for_day(_today() - timedelta(days=2), cutoff_time=AUTOCLOSE_CUTOFF)
            mgr.autoclose_for_day(_today() - timedelta(days=3), cutoff_time=AUTOCLOSE_CUTOFF)
        except Exception as e:
            print(f"[BOOT] Autoclose catch-up failed: {e}")

    # One-time backfill at boot (safe regardless of scheduler)
    _backfill_weekly_if_missing(mgr)
    _backfill_monthly_if_missing(mgr)

    # Start repair monitor (auto-correct time and auto-start scheduler when sane)
    threading.Thread(target=_clock_repair_monitor, daemon=True).start()

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
        pass
    finally:
        try:
            lcd.clear()
        except Exception:
            pass
        GPIO.cleanup()
        try:
            picam2.stop()
        except Exception:
            pass
        print("[SYSTEM] Shutdown")
        
        
#-------future zero byte repair scripts-------------------

def _sanitize_attendance_folder():
    for fname in os.listdir(ATTEND_DIR):
        if fname.endswith(".csv"):
            path = os.path.join(ATTEND_DIR, fname)
            try:
                if os.path.getsize(path) == 0:
                    print(f"[SANITY] Reinitializing empty file {path}")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(",".join(DAILY_HEADERS) + "\n")
            except Exception as e:
                print(f"[SANITY] Could not stat {path}: {e}")


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
                     "SUCCESSFULLY LOGGED"])
            time.sleep(2.2)
            GPIO.output(GREEN_LED_PIN if action == "Check-In" else RED_LED_PIN, GPIO.LOW)
        else:
            display(["LOG FAILED", "TRY AGAIN"])
    else:
        display(["FACE NOT", "RECOGNIZED"])
    time.sleep(1.5)
    display(get_greeting_lines())
    transition_to("IDLE")

if __name__ == "__main__":
    main()
