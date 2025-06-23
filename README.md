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
- Developed Raspberry Pi face recognition app
- Integrated PostgreSQL logging
- Built auto reporting + offline sync
- Writing hardware documentation & deployment guide

---

## Technologies used
- Raspberry Pi 4
- PiCamera
- Python 3.11
- OpenCV + face_recognition
- APScheduler (auto reporting)
- FastAPI (backend integration)
- PostgreSQL (data storage)
- LCD + I2C
- RPi.GPIO
