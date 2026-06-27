# Vanguard AI: Advanced Traffic Control & Biometric Security Suite

[![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/AI%20Core-YOLOv8%20%7C%20InsightFace-orange.svg)]()
[![Database](https://img.shields.io/badge/Database-SQLite3%20%2B%20AES--256-green.svg)]()
[![UI](https://img.shields.io/badge/UI-Tkinter%20%7C%20ttkbootstrap-purple.svg)]()

Vanguard is an enterprise-grade, high-performance automated access control and identity verification suite. Designed for high-throughput institutional gates (e.g., university campuses), Vanguard combines real-time Deep Learning pipelines with military-grade biometric encryption and robust network camera telemetry to deliver a secure, localized, and standalone surveillance solution.

The system is split into two specialized applications sharing a synchronized, multi-threaded secure data layer:
1. **Vanguard Monitoring System (`main.py`)**: The primary real-time pipeline for plate recognition, facial tracking, and automated entry/denial execution.
2. **Vanguard Enrollment System (`enrollment.py`)**: A secure administrative terminal featuring multi-view biometric enrollment and dynamic identity profiling.

---

## 🛠️ Key Architectural Pillars

### 1. Dual-Modal Neural Inference Pipeline
*   **License Plate Recognition (LPR)**: Powered by a split-stage YOLO architecture. The first model isolates the vehicle plate zone (`vanguard_plate_v2.pt`), while a highly optimized secondary model (`vanguard_char_v3.pt`) executes character segmentation and alphanumeric token parsing.
*   **Facial Analysis & Tracking**: Integrates the `buffalo_s` model family via InsightFace over ONNX Runtime. It uses a thread-safe custom Centroid Tracker (`FaceTracker`) with identity stabilization to track moving targets, eliminating duplicate logs and fluctuating similarity scores.
*   **FPS Optimization Core**: Multi-threaded execution pipelines decoupling the frame acquisition thread (RTSP/Webcam) from the heavy AI inference loop using high-throughput atomic queues (`queue.Queue`). Includes a spatial throttling option allowing customizable frame skips without losing tracking persistence.

### 2. Biometric Cryptography & Secure Storage
*   **AES-256 Symmetric Encryption**: Facial embeddings are serialized via `pickle` and fully encrypted using the Fernet specification (`cryptography.fernet`) before being committed to the database as high-security binary BLOBs. This effectively mitigates raw biometric data theft vectors even in compromised environment scenarios.
*   **Isolated Schema Engine**: Built upon a single-connection, thread-safe SQLite3 wrapper utilizing Write-Ahead Logging (`WAL`), low-overhead memory temporary storage allocation, and automated multi-layer table migrations.

### 3. Localization & Temporal Access Controls
*   **Jalali (Persian Shami) Calendar Integration**: Completely replaces traditional chronological countdowns with absolute temporal access restrictions. The system converts Persian inputs directly to global Unix timestamps for verification, checking expiration limits down to the second.
*   **Bidi Real-time Rendering**: Combines `arabic_reshaper` and `python-bidi` with custom TrueType graphics subroutines to overlay complex right-to-left Persian text on real-time BGR frame spaces flawlessly.

### 4. Production Hardening & Anti-Tamper Hooks
*   **Dynamic UI Integrity Inspection**: Incorporates embedded obfuscated base64 integrity strings and dynamic logical runtime callbacks that verify visual attribution state. Any unauthorized removal or alteration of development credentials results in automated application termination (`sys.exit()`).
*   **Resilient Camera Telemetry**: Includes advanced stream telemetry overriding native FFmpeg behaviors over RTSP. By forcing TCP transport protocols over UDP and clearing buffers natively, it guarantees zero-latency, artifact-free processing of enterprise network streams (e.g., Dahua, Hikvision, Tiandy IP Cameras).

---

## 🏗️ System Architecture & Workflow

```
[ Camera Stream / RTSP over TCP ] ──► [ Atomic Frame Queue ]
│
┌───────────────────────┴───────────────────────┐
▼                                               ▼
[ Deep LPR Pipeline ]                           [ InsightFace Tracking ]
(Plate Detection ──► Char OCR)                   (Centroid Fixation ──► Identity Sync)
│                                               │
└───────────────────────┬───────────────────────┘
▼
[ Synchronized Safe Core ]
│
┌─────────────────────────┴─────────────────────────┐
▼                                                   ▼
[ Permission Engine (Unix TS) ]                     [ AES-256 Encrypted BLOB Engine ]
Evaluates Temporal Expiry Bound                    Decrypts Embedding Matrix for Cosine Match
```

---

## 🗃️ Database Schema Spec

The database engine (`database.py`) maintains absolute relational integrity across 5 operational environments:

| Table Name | Primary Key | Key Columns | Cryptographic Status |
| :--- | :--- | :--- | :--- |
| `permissions` | `plate_number` | `owner_name`, `role`, `max_duration (Unix TS)` | Plaintext Normalized Token |
| `traffic_logs` | `id (Auto-Inc)` | `plate_number`, `status`, `image_path`, `detection_time` | Relational Storage Lookup |
| `vehicles_inside` | `plate_number` | `entry_time`, `owner_name` | Runtime Live Cache |
| `face_users` | `national_id` | `name`, `role`, `face_encoding (BLOB)`, `created_at` | **AES-256 Fernet Encrypted** |
| `face_logs` | `id (Auto-Inc)` | `person_name`, `role`, `status`, `image_path`, `detection_time`| Isolation Tracking Index |

---

## 🚀 Installation & Local Workspace Setup

### Prerequisites
*   Windows 10 / 11 (Architecture optimized for AMD64/x86_64)
*   Python 3.9.x
*   CUDA Toolkit & cuDNN (Optional, for GPU acceleration via `CUDAExecutionProvider`)

### 1. Clone the Repository
```bash
git clone https://github.com/Mousapour-Lab/Vanguard-AI-Suite.git
cd Vanguard-AI-Suite
```

### 2. Instantiate and Activate Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Deploy Dependency Tree

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

*Note: If building on offline environments, ensure the `.insightface/models/buffalo_s` weights path is accurately provisioned inside your user root partition.*

---

## 📦 Compilation & Production Deployment

The project features an automated, single-command orchestration engine (`build.py`) that handles dynamic `_internal` library compilation via PyInstaller, bundles binary assets, and outputs a complete desktop application suite.

### To compile the binary runtime distribution:

```powershell
python build.py
```

The unified production build is allocated directly inside: `.\dist\Vanguard_Suite\`

### Installation Wizard Compilation (Inno Setup)

An optimized script (`setup.iss`) is included to compile the installation layout into a high-speed, enterprise desktop wizard utilizing `lzma2/fast` architecture:

1. Open Inno Setup Compiler.
2. Load `setup.iss`.
3. Execute Compile (`Ctrl + F9`).
4. Your clean distribution package `Vanguard_Setup_Fast.exe` will be built inside a dedicated `Vanguard_Build` directory right on your Desktop.

---

## 🪪 Developer & Maintainer

* **Lead Engineer**: Mohammad Mousapour
* **Role**: Computer Engineering Student & Software Automations Developer
* **GitHub Profile**: [@Mousapour-Lab](https://github.com/Mousapour-Lab)

---

## 📄 License & Intellectual Property

Copyright © 2026 Mohammad Mousapour. All rights reserved.
Developed as a high-security automated computer vision suite. Unauthorized commercial redistribution, binary decompilation, or brand removal is strictly protected under local property policies.

---
<div dir="rtl">

## سامانه هوشمند کنترل تردد دانشگاهی و امنیت بیومتریک ونگارد (Vanguard AI)
پروژه ونگارد (Vanguard) یک سوئیت نرم‌افزاری تجاری، با کارایی بالا و بومی‌سازی شده برای کنترل تردد خودکار و تایید هویت بیومتریک است. این سامانه با هدف استفاده در گیت‌های پرتردد سازمانی و دانشگاهی طراحی شده و با تلفیق خطوط لوله یادگیری عمیق (Deep Learning)، رمزنگاری داده‌های بیومتریک در سطح نظامی و تلمتری پایدار دوربین‌های تحت شبکه، یک راهکار نظارتی کاملاً محلی (Offline) و مستقل ارائه می‌دهد.

این سامانه از دو بخش نرم‌افزاری مجزا تشکیل شده که هر دو از یک لایه دیتابیس امن و همگام‌سازی شده (Thread-Safe) استفاده می‌کنند:
*   **سامانه مانیتورینگ و کنترل تردد اصلی (`main.py`)**: هسته پردازش تصویر زنده جهت شناسایی پلاک خودرو، ردیابی چهره و صدور خودکار مجوز ورود/خروج.
*   **سامانه ثبت‌نام و مدیریت پرسنل (`enrollment.py`)**: پنل اداری امن جهت اسکن سه‌بعدی و چندزاویه‌ای چهره و تعریف دقیق سطوح دسترسی پرسنل.

---

### 🛠️ ستون‌های اصلی معماری پروژه
#### ۱. خط لوله استنتاج عصبی دوگانه (Dual-Modal Inference)
*   **تشخیص و خوانش پلاک (LPR)**: بهره‌گیری از معماری دو مرحله‌ای YOLO. مدل اول پلاک خودرو را در تصویر ایزوله می‌کند (`vanguard_plate_v2.pt`) و مدل دوم (`vanguard_char_v3.pt`) با دقت بالا عملیات بخش‌بندی کاراکترها و تبدیل آن‌ها به توکن‌های متنی انگلیسی را انجام می‌دهد.
*   **تحلیل و ردیابی چهره**: ادغام مدل‌های خانواده `buffalo_s` از فریم‌ورک InsightFace بر بستر ONNX Runtime. ردیاب هوشمند اختصاصی (`FaceTracker`) با استفاده از الگوریتم Centroid و پایدارسازی هویت، مانع از ثبت لاگ‌های تکراری یا نوسان درصد شباهت در تصاویر متحرک می‌شود.
*   **بهینه‌سازی نرخ فریم (FPS Throttling)**: جداسازی کامل ترد دریافت تصاویر (وبکم/RTSP) از ترد پردازش‌های سنگین هوش مصنوعی با کمک صف‌های اتمیک (`queue.Queue`) که پردازش بدون لگ را روی پردازنده‌های معمولی (CPU) تضمین می‌کند.

#### ۲. امنیت اطلاعات و رمزنگاری بیومتریک در سطح نظامی
*   **رمزنگاری متقارت AES-256**: امبدینگ‌های ۵۱۲ بعدی چهره ابتدا توسط `pickle` سریالایز شده و سپس با الگوریتم استاندارد Fernet (`cryptography.fernet`) رمزنگاری می‌شوند تا به صورت باینری (BLOB) در دیتابیس ذخیره گردند. این کار مانع از سرقت داده‌های بیومتریک حتی در صورت دسترسی غیرمجاز به فایل دیتابیس می‌شود.
*   **دیتابیس ایزوله و همگام‌سازی شده**: پیاده‌سازی کلاس مدیریت دیتابیس به صورت کاملاً Thread-Safe با استفاده از قفل‌های بازگشتی (RLock) در پایتون، فعال‌سازی حالت نوشتن موازی دیتابیس (WAL) و بهینه‌سازی سرعت تراکنش‌ها.

#### ۳. بومی‌سازی و موتور رندر فارسی
*   **یکپارچه‌سازی تقویم جلالی (شمسی)**: حذف کامل سیستم‌های تایمر سنتی و جایگزینی آن با سیستم کنترل دسترسی بر اساس تاریخ انقضای دقیق شمسی. نرم‌افزار به صورت خودکار ورودی‌های تاریخ شمسی کاربر را به Timestamp استاندارد یونیکس تبدیل کرده و دسترسی‌ها را ثانیه‌به‌ثانیه کنترل می‌کند.
*   **رندر بی‌نقص متون فارسی (Bidi)**: تلفیق کتابخانه‌های `arabic_reshaper` و `python-bidi` برای نوشتن راست‌به‌چپ متون فارسی روی فریم‌های زنده تصویر با فونت‌های TrueType بومی بدون به هم ریختگی حروف.

#### ۴. مقاوم‌سازی نرم‌افزار و قلاب‌های ضدتخریب (Anti-Tamper)
*   **بررسی یکپارچگی رابط کاربری**: تعبیه لایه‌های امنیتی با کدهای رمزنگاری‌شده Base64 که در پس‌زمینه صحت کپی‌رایت توسعه‌دهنده را بررسی می‌کنند. در صورت هرگونه دستکاری یا تغییر در کدهای کپی‌رایت، نرم‌افزار بلافاصله به صورت خودکار خاموش می‌شود (`sys.exit()`).
*   **تلمتری پایدار جریان ویدیو**: اعمال تنظیمات اختصاصی روی هک درایور FFmpeg برای دوربین‌های تحت شبکه (مانند Dahua). سیستم با اجبار پروتکل RTSP روی بستر TCP به جای UDP و پاکسازی بلادرنگ بافرها، استریم بدون تاخیر و فاقد آرتیفکت‌های تصویری را ارائه می‌دهد.

---

### 🏗️ معماری سیستم و گردش کار
<div dir="ltr">

```
[ دوربین مداربسته / RTSP روی TCP ] ──► [ صف اتمیک فریم‌ها ]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         [ خط لوله پلاک‌خوان هوشمند ]                       [ ردیابی چهره InsightFace ]
       (یافتن پلاک ──► پردازش کاراکترها)               (الگوریتم سنتروئید ──► تایید هویت)
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                                   [ هسته پردازش همگام ]
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
       [ موتور مجوزدهی (Unix TS) ]                     [ هسته رمزنگاری AES-256 ]
    بررسی تاریخ انقضای پلاک به ثانیه                  رمزگشایی امبدینگ چهره برای تطبیق کوساین
```
</div>

---

### 🗃️ طرحواره (Schema) دیتابیس امن
دیتابیس سیستم (`database.py`) یکپارچگی اطلاعات را در ۵ جدول عملیاتی حفظ می‌کند:

| نام جدول | کلید اصلی | ستون‌های کلیدی | وضعیت رمزنگاری |
| :--- | :--- | :--- | :--- |
| `permissions` | `plate_number` | `owner_name`, `role`, `max_duration (Unix TS)` | توکن پلاک نرمال شده |
| `traffic_logs` | `id (Auto-Inc)` | `plate_number`, `status`, `image_path`, `detection_time` | اطلاعات تردد خودروها |
| `vehicles_inside` | `plate_number` | `entry_time`, `owner_name` | کش خودروهای فعال داخل دانشگاه |
| `face_users` | `national_id` | `name`, `role`, `face_encoding (BLOB)`, `created_at` | **رمزنگاری شده با کلید اختصاصی AES-256** |
| `face_logs` | `id (Auto-Inc)` | `person_name`, `role`, `status`, `image_path`, `detection_time`| اطلاعات تردد افراد |

---

### 🚀 راهنمای نصب و راه‌اندازی محلی
#### پیش‌نیازها
*   ویندوز 10 یا 11 (معماری پردازنده x64)
*   پایتون نسخه 3.9.x
*   ابزار CUDA Toolkit و cuDNN (اختیاری - جهت شتاب‌دهی سخت‌افزاری هوش مصنوعی با کارت‌های گرافیک انویدیا)

#### ۱. شبیه‌سازی (Clone) مخزن
```bash
git clone https://github.com/Mousapour-Lab/Vanguard-AI-Suite.git
cd Vanguard-AI-Suite
```

#### ۲. ساخت و فعال‌سازی محیط مجازی پایتون
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### ۳. نصب وابستگی‌های پکیج
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
*نکته: در صورت استفاده از سیستم‌های کاملاً آفلاین، مطمئن شوید که پوشه مدل‌های چهره در مسیر `.insightface/models/buffalo_s` در پوشه کاربری شما کپی شده باشد.*

---

### 📦 کامپایل پروژه و ساخت نسخه نصبی تک‌فایلی
پروژه مجهز به یک اسکریپت کاملاً خودکار برای کامپایل آسان به نام `build.py` است. این اسکریپت تمام کتابخانه‌های `_internal` و لایه‌های گرافیکی هر دو نرم‌افزار را ادغام کرده و یک نسخه پرتابل بی‌نقص خروجی می‌دهد.

##### دستور کامپایل در پاورشل:
```powershell
python build.py
```
خروجی نهایی و یکپارچه در این پوشه ذخیره خواهد شد: `.\dist\Vanguard_Suite\`

#### ساخت جادوگر نصب (Inno Setup)
یک اسکریپت اختصاصی (`setup.iss`) با معماری فشرده‌سازی پرسرعت `lzma2/fast` برای SSDها طراحی شده است:
1.  نرم‌افزار Inno Setup Compiler را باز کنید.
2.  فایل `setup.iss` را لود کنید.
3.  دکمه کامپایل (`Ctrl + F9`) را بزنید.
4.  فایل نصبی نهایی شما به نام `Vanguard_Setup_Fast.exe` در پوشه `Vanguard_Build` روی دسکتاپ شما ساخته می‌شود.

---

### 🪪 توسعه‌دهنده و صاحب اثر
*   **مهندس ناظر و برنامه‌نویس**: محمد موسی‌پور
*   **سمت**: دانشجوی مهندسی کامپیوتر و توسعه‌دهنده اتوماسیون‌های نرم‌افزاری هوشمند
*   **پروفایل گیت‌هاب**: [@Mousapour-Lab](https://github.com/Mousapour-Lab)

---

### 📄 حق مالکیت معنوی و لایسنس
کلیه حقوق مادی و معنوی این نرم‌افزار متعلق به محمد موسی‌پور می‌باشد. این سیستم به عنوان یک سوئیت پیشرفته بینایی ماشین با تدابیر امنیتی و بومی توسعه یافته است. هرگونه بازنشر تجاری، مهندسی معکوس کدهای کامپایل شده یا حذف برند و کپی‌رایت توسعه‌دهنده بدون کسب اجازه کتبی ممنوع و تحت پیگرد قانونی است.

</div>
