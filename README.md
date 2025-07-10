# EMSATS - Raspberry Pi Employee Attendance System

This project is the **hardware-side implementation** of the **Employee Management System for Asataura Technology Services (EMSATS)**.

It is an **offline-first Raspberry Pi-based employee attendance system**, which uses:
- **Facial Recognition**
- **LCD Display**
- **Check-in / Check-out buttons**
- **LED indicators**
- **PostgreSQL database**

✅ The system works autonomously in the office — it recognizes employees via face recognition, logs attendance, and auto-generates **weekly and monthly reports**.  
✅ If offline, it stores records locally and syncs automatically once connected to the EMSATS central backend.

This project is part of a larger EMSATS platform (with FastAPI backend and React frontend), providing a **modern, scalable attendance solution** for the company.

---

## My Role
- Lead Hardware Developer for EMSATS
- Developed Raspberry Pi face recognition application
- Integrated PostgreSQL attendance logging with offline-first fallback
- Built auto-reporting & background sync
- Authored hardware deployment and integration documentation

---

## Technologies Used
- Raspberry Pi 4
- PiCamera (with Picamera2)
- Python 3.11
- OpenCV + face_recognition
- APScheduler (background jobs)
- PostgreSQL (central and edge database)
- FastAPI (backend integration)
- LCD (16x2 I2C)
- RPi.GPIO for buttons + LED logic

---

## Requirements
- Python packages:
  - face_recognition
  - opencv-python
  - pandas
  - psycopg2
  - python-dotenv
  - RPi.GPIO
  - RPLCD
  - picamera2

##  Setup

1. **Install Dependencies**
```bash
sudo apt update
sudo apt install libjpeg-dev libatlas-base-dev python3-pip
pip install -r requirements.txt
```

2. **Setup Environment Variables**
```bash
cp .env.example .env
nano .env
```

3. **Enable Raspberry Pi Interfaces**
```bash
sudo raspi-config
# Enable I2C and Camera
```

4. **GPIO Pin Map**
| Purpose     | GPIO Pin |
|-------------|----------|
| Check-In    | GPIO 22  |
| Check-Out   | GPIO 23  |
| Green LED   | GPIO 27  |
| Red LED     | GPIO 17  |

5. **Run the System**
```bash
python3 main.py
```

---

## 🗃️ PostgreSQL Schema
```sql
CREATE TABLE employees (
  employee_id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  image_url TEXT
);

CREATE TABLE employee_attendance (
  id SERIAL PRIMARY KEY,
  name TEXT,
  employee_id UUID,
  type TEXT,
  date DATE,
  time TEXT,
  timestamp DOUBLE PRECISION
);
```

---

## 🚀 Key Features
- ✅ Real-time facial recognition-based check-in / check-out
- ✅ Offline-first sync with automatic retry queue
- ✅ Periodic background sync of facial encodings from image URLs
- ✅ Modular logging architecture using `AttendanceManager`
- ✅ Exportable `.xlsx` reports from PostgreSQL logs

---

## 📤 Export Logs
Run the export manually:
```python
AttendanceManager().export_attendance()
```
File saved to:
```
/home/pi/Desktop/PROJECT/N-facial-recognition-QRCODE/exported_attendance.xlsx
```

---

## 🚀 Deployment (systemd)
Create a unit file:
```ini
[Unit]
Description=EMSATS Facial Recognition Attendance Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/emsats/main.py
WorkingDirectory=/home/pi/emsats
Restart=always

[Install]
WantedBy=multi-user.target
```
Enable on boot:
```bash
sudo systemctl enable emsats.service
```

---

## 🔐 Security Best Practices
- Never hardcode credentials; use `.env`
- Add `.env` to `.gitignore`
- Distributed `.env.example` to teammates

---

## 🤝 Acknowledgements
- `face_recognition` by Adam Geitgey
- Raspberry Pi Foundation (Picamera2)
- `RPLCD` for LCD I2C driver
- `python-dotenv` for secure config loading

---

**EMSATS is a production-grade, modular attendance tracking system deployed across Asataura Technology offices.**
Scalable to include QR recognition, mobile API integrations, and real-time dashboards.
