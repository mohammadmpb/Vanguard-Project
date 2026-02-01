Markdown
# 🛡️ Vanguard AI Security System

> **Advanced Surveillance System with Real-time License Plate & Face Recognition**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLO](https://img.shields.io/badge/AI-YOLOv11-green)
![InsightFace](https://img.shields.io/badge/Biometrics-InsightFace-orange)
![Security](https://img.shields.io/badge/Security-Encrypted%20DB-red)

## 📖 Overview
**Vanguard** is a comprehensive, AI-powered security monitoring system designed for universities and restricted areas. It combines computer vision state-of-the-art models to control traffic and human access in real-time.

Unlike simple detection scripts, Vanguard features a **secure encrypted database**, a modern GUI, and optimized performance for real-time processing.

## ✨ Key Features

### 🚗 Intelligent Traffic Control (LPR)
- **Engine:** Powered by custom-trained **YOLO models** (vanguard_plate_v2.pt).
- **Capabilities:** High-accuracy detection of **Persian License Plates**.
- **OCR:** Custom character recognition model for extracting numbers and city codes.
- **Logging:** Automatic entry/exit logging with timestamps and owner identification.

### 👤 Biometric Access Control
- **Engine:** Utilizes **InsightFace** (Buffalo_S) for military-grade face recognition.
- **Enrollment:** Secure enrollment process with multi-angle face scanning.
- **Tracking:** Real-time face tracking and identification overlay.

### 🔐 Security & Architecture
- **Database Encryption:** All sensitive biometric data (face embeddings) are encrypted using **Fernet (Cryptography)** before storage.
- **GameGuard Protocol:** Intelligent directory filtering to prevent scanning game/system files.
- **Modern UI:** Built with `ttkbootstrap` for a clean, dark-mode interface.

## 🛠️ Tech Stack
- **Core:** Python 3.x
- **Vision:** OpenCV, Ultralytics YOLO, InsightFace
- **Data:** SQLite3 (Encrypted), NumPy
- **GUI:** Tkinter, Ttkbootstrap

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mohammadmpb/Vanguard-Project.git](https://github.com/mohammadmpb/Vanguard-Project.git)
   cd Vanguard-Project
Install dependencies:

Bash
pip install -r requirements.txt
Run the System:

For Monitoring:

Bash
python main.py
For New User Enrollment:

Bash
python enrollment.py
📂 Project Structure
Vanguard-Project/
├── weights/           # Trained AI Models (.pt)
├── logs/              # Detection snapshots & logs
├── database.py        # Secure Database Manager
├── main.py            # Main Application (GUI)
├── enrollment.py      # Face Registration Tool
└── requirements.txt   # Dependencies
👨‍💻 Author
Developed by Mohammad. Computer Engineering Student at Mohajer University.