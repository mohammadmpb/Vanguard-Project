import sqlite3
import datetime
import pickle
import os
import numpy as np
from cryptography.fernet import Fernet


class DatabaseManager:
    def __init__(self, db_name="vanguard_secure.db", key_file="vanguard.key"):
        """
        مدیریت دیتابیس با قابلیت رمزنگاری نظامی.
        اگر فایل کلید وجود نداشته باشد، می‌سازد.
        """
        # --- تغییر مهم: پیدا کردن آدرس دقیق پوشه‌ای که فایل database.py در آن است ---
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # --- تغییر مهم: چسباندن آدرس پوشه به اسم فایل‌ها ---
        self.db_name = os.path.join(base_dir, db_name)
        self.key_file = os.path.join(base_dir, key_file)

        # 1. لود یا ساخت کلید امنیتی
        self.key = self._load_or_generate_key()
        self.cipher = Fernet(self.key)

        # 2. اتصال به دیتابیس (استفاده از آدرس کامل)
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def _load_or_generate_key(self):
        """مدیریت کلید رمزنگاری (مشترک بین تمام ماژول‌ها)"""
        if os.path.exists(self.key_file):
            with open(self.key_file, "rb") as kf:
                return kf.read()
        else:
            print("⚠ New Security Key Generated!")
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as kf:
                kf.write(key)
            return key

    def _encrypt_data(self, data):
        """تبدیل داده پایتون به بایت‌های رمزگذاری شده"""
        try:
            pickled = pickle.dumps(data)
            return self.cipher.encrypt(pickled)
        except Exception as e:
            print(f"Encryption Error: {e}")
            return None

    def _decrypt_data(self, encrypted_data):
        """رمزگشایی و بازگرداندن داده اصلی"""
        try:
            decrypted = self.cipher.decrypt(encrypted_data)
            return pickle.loads(decrypted)
        except Exception as e:
            # اگر کلید اشتباه باشد یا داده خراب باشد
            return None

    def create_tables(self):
        # --- جدول مجوزهای پلاک (بدون تغییر) ---
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS permissions
                            (
                                plate_number
                                TEXT
                                PRIMARY
                                KEY,
                                owner_name
                                TEXT,
                                role
                                TEXT,
                                max_duration
                                INTEGER
                                DEFAULT
                                0,
                                created_at
                                TIMESTAMP
                                DEFAULT
                                CURRENT_TIMESTAMP
                            )
                            """)

        # --- جدول لاگ تردد خودرو (بدون تغییر) ---
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS traffic_logs
                            (
                                id
                                INTEGER
                                PRIMARY
                                KEY
                                AUTOINCREMENT,
                                plate_number
                                TEXT,
                                status
                                TEXT,
                                image_path
                                TEXT,
                                detection_time
                                TIMESTAMP
                                DEFAULT
                                CURRENT_TIMESTAMP
                            )
                            """)

        # --- جدول خودروهای داخل (بدون تغییر) ---
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS vehicles_inside
                            (
                                plate_number
                                TEXT
                                PRIMARY
                                KEY,
                                entry_time
                                TIMESTAMP,
                                owner_name
                                TEXT
                            )
                            """)

        # --- جدول کاربران تشخیص چهره (تغییر یافته برای امنیت) ---
        # نکته: face_encoding اینجا داده رمزگذاری شده (BLOB) را نگه می‌دارد
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS face_users
                            (
                                national_id
                                TEXT
                                PRIMARY
                                KEY,
                                name
                                TEXT,
                                role
                                TEXT,
                                face_encoding
                                BLOB,
                                created_at
                                TIMESTAMP
                                DEFAULT
                                CURRENT_TIMESTAMP
                            )
                            """)

        # --- جدول آمار تردد انبوه ---
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS traffic_stats
                            (
                                id
                                INTEGER
                                PRIMARY
                                KEY
                                AUTOINCREMENT,
                                log_date
                                TEXT,
                                log_time
                                TEXT,
                                registered_count
                                INTEGER
                                DEFAULT
                                0,
                                unknown_count
                                INTEGER
                                DEFAULT
                                0,
                                total_count
                                INTEGER
                                DEFAULT
                                0
                            )
                            """)
        self.conn.commit()

    # ==========================
    # بخش پلاک‌خوان (License Plate)
    # ==========================
    def add_permission(self, plate, owner, role, duration):
        try:
            self.cursor.execute(
                "INSERT INTO permissions (plate_number, owner_name, role, max_duration) VALUES (?, ?, ?, ?)",
                (plate, owner, role, duration))
            self.conn.commit()
            return True
        except:
            return False

    def delete_permission(self, plate):
        self.cursor.execute("DELETE FROM permissions WHERE plate_number=?", (plate,))
        self.conn.commit()

    def check_permission(self, plate):
        self.cursor.execute("SELECT owner_name, role, max_duration FROM permissions WHERE plate_number=?", (plate,))
        res = self.cursor.fetchone()
        if res: return True, res[0], res[1], res[2]
        return False, "Unknown", "Visitor", 0

    def get_all_permissions(self):
        self.cursor.execute("SELECT plate_number, owner_name, role, max_duration FROM permissions")
        return self.cursor.fetchall()

    def log_entry(self, plate, status, path, owner="Unknown"):
        # جلوگیری از لاگ تکراری زیر 2 دقیقه
        self.cursor.execute("SELECT detection_time FROM traffic_logs WHERE plate_number=? ORDER BY id DESC LIMIT 1",
                            (plate,))
        last = self.cursor.fetchone()
        if last:
            last_time = datetime.datetime.strptime(last[0], "%Y-%m-%d %H:%M:%S")
            if (datetime.datetime.now() - last_time).total_seconds() < 120:
                return False

        self.cursor.execute("INSERT INTO traffic_logs (plate_number, status, image_path) VALUES (?, ?, ?)",
                            (plate, status, path))

        if status == "Allowed":
            self.cursor.execute(
                "INSERT OR REPLACE INTO vehicles_inside (plate_number, entry_time, owner_name) VALUES (?, ?, ?)",
                (plate, datetime.datetime.now().strftime("%H:%M:%S"), owner))
        self.conn.commit()
        return True

    def get_all_logs(self):
        self.cursor.execute("SELECT * FROM traffic_logs ORDER BY id DESC LIMIT 200")
        return self.cursor.fetchall()

    def get_vehicles_inside(self):
        self.cursor.execute("SELECT * FROM vehicles_inside")
        return self.cursor.fetchall()

    def mark_exit(self, plate):
        self.cursor.execute("DELETE FROM vehicles_inside WHERE plate_number=?", (plate,))
        self.conn.commit()

    def clear_database(self):
        self.cursor.execute("DELETE FROM traffic_logs")
        self.cursor.execute("DELETE FROM vehicles_inside")
        self.cursor.execute("DELETE FROM traffic_stats")
        self.conn.commit()

    # ==========================
    # بخش تشخیص چهره امن (Secure Face Recognition)
    # ==========================
    def add_face_user(self, nid, name, role, encoding_array):
        """ذخیره کاربر با رمزنگاری داده‌های بیومتریک"""
        try:
            # داده‌ها اول رمزگذاری می‌شوند، سپس ذخیره می‌شوند
            encrypted_blob = self._encrypt_data(encoding_array)
            if encrypted_blob is None:
                raise ValueError("Encryption failed")

            self.cursor.execute("""
                INSERT OR REPLACE INTO face_users (national_id, name, role, face_encoding)
                VALUES (?, ?, ?, ?)
            """, (nid, name, role, encrypted_blob))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"DB Face Error: {e}")
            return False

    def get_all_faces(self):
        """بازیابی و رمزگشایی چهره‌ها برای شناسایی"""
        self.cursor.execute("SELECT national_id, name, role, face_encoding FROM face_users")
        rows = self.cursor.fetchall()

        known_encodings = []
        known_names = []
        known_roles = []
        known_ids = []  # برای استفاده احتمالی در آینده

        for row in rows:
            nid, name, role, encrypted_blob = row

            # رمزگشایی داده
            decrypted_encoding = self._decrypt_data(encrypted_blob)

            if decrypted_encoding is not None:
                # اگر فرمت لیست بود (از سیستم مالتی ویو)، به لیست اصلی اضافه کن
                if isinstance(decrypted_encoding, list):
                    # در اینجا ما همه وکتورها را به عنوان رفرنس برای یک نفر در نظر می‌گیریم
                    # اما چون فعلا ساختار لیست‌های موازی داریم، اولین وکتور را به عنوان نماینده برمی‌داریم
                    # یا (بهتر): منطق تشخیص چهره باید تغییر کند تا از لیست پشتیبانی کند.
                    # برای سازگاری با کد فعلی تو، ما فعلا اولین وکتور لیست را برمی‌داریم:
                    if len(decrypted_encoding) > 0 and isinstance(decrypted_encoding[0], np.ndarray):
                        known_encodings.append(decrypted_encoding[0])  # فعلا سینگل شات
                    else:
                        known_encodings.append(decrypted_encoding)
                else:
                    known_encodings.append(decrypted_encoding)

                known_names.append(name)
                known_roles.append(role)
                known_ids.append(nid)

        return known_encodings, known_names, known_roles

    # ==========================
    # بخش آمار (Statistics)
    # ==========================
    def log_traffic_batch(self, reg_count, unk_count):
        if reg_count == 0 and unk_count == 0: return

        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M")
        total = reg_count + unk_count

        try:
            self.cursor.execute("""
                                INSERT INTO traffic_stats (log_date, log_time, registered_count, unknown_count, total_count)
                VALUES (?, ?, ?, ?, ?)
            """, (date_str, time_str, reg_count, unk_count, total))
            self.conn.commit()
            # print(f"📊 Traffic Saved: {total}")
        except Exception as e:
            print(f"DB Log Error: {e}")

    def close(self):
        if self.conn:
            self.conn.close()