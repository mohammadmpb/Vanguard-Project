import os
import pickle
import json
import cv2
import re
import time
import queue
import datetime
import threading
import numpy as np
from typing import Optional

# --- Third Party Imports ---
from PIL import Image, ImageTk, ImageFont, ImageDraw
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import arabic_reshaper
from bidi.algorithm import get_display
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import tkinter.ttk as original_ttk

# --- Local Imports ---
from database import DatabaseManager


# ==========================================
#              تنظیمات و ثابت‌ها
# ==========================================
os.environ["YOLO_VERBOSE"] = "False"
FACE_THRESHOLD = 0.55
TRACK_DISTANCE = 80
MAX_MISSING_FRAMES = 20

PERSIAN_CHAR_MAP = {
    'الف': 'Alef', 'ب': 'B', 'پ': 'P', 'ت': 'T', 'ث': 'Se', 'ج': 'J', 'چ': 'Che',
    'ح': 'H', 'خ': 'Khe', 'د': 'D', 'ذ': 'Z', 'ر': 'R', 'ز': 'Z', 'ژ': 'Zh',
    'س': 'S', 'ش': 'Sh', 'ص': 'Sad', 'ض': 'Zad', 'ط': 'Ta', 'ظ': 'Za', 'ع': 'Ein',
    'غ': 'Gh', 'ف': 'F', 'ق': 'Gh', 'ک': 'K', 'گ': 'G', 'ل': 'L', 'م': 'M',
    'ن': 'N', 'و': 'V', 'ه': 'H', 'ی': 'Y', '♿': 'Malul'
}
PERSIAN_NUM_MAP = {'۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4', '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'}
TRANSLATION_MAP = {
    '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴', '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹',
    'B': 'ب', 'J': 'ج', 'D': 'د', 'S': 'س', 'Sad': 'ص', 'Ta': 'ط', 'Gh': 'ق', 'L': 'ل', 'M': 'م',
    'N': 'ن', 'V': 'و', 'H': 'ه', 'Y': 'ی', 'T': 'ت', 'Ein': 'ع', 'P': 'پ', 'Se': 'ث', 'Z': 'ز',
    'Sh': 'ش', 'K': 'ک', 'G': 'گ', 'Alef': 'الف', 'Malul': '♿',
    'R': 'ر', 'Che': 'چ', 'Khe': 'خ', 'Zh': 'ژ', 'Zad': 'ض', 'Za': 'ظ', 'F': 'ف'
}
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'J', 'D', 'S', 'Sad', 'Ta', 'Gh',
               'L', 'M', 'N', 'V', 'H', 'Y', 'T', 'Ein', 'P', 'Se', 'Z', 'Sh', 'K', 'G', 'Alef', 'Malul']


# ==========================================
#        کلاس ردیاب هوشمند (Face Tracker)
# ==========================================
class FaceTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}  # {id: {'bbox': box, 'missing': 0, 'name': name, 'role': role, 'counted': False}}

    def update(self, detections):
        """
        detections: list of {'bbox': [x1,y1,x2,y2], 'name': str, 'role': str}
        """
        updated_tracks = {}

        new_centers = []
        for det in detections:
            box = det['bbox']
            cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            new_centers.append((cx, cy))

        track_ids = list(self.tracks.keys())
        track_centers = []
        for tid in track_ids:
            box = self.tracks[tid]['bbox']
            cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            track_centers.append((cx, cy))

        if new_centers and track_centers:
            dists = np.zeros((len(track_ids), len(new_centers)))
            for t, tc in enumerate(track_centers):
                for n, nc in enumerate(new_centers):
                    dists[t, n] = np.linalg.norm(np.array(tc) - np.array(nc))

            rows = dists.min(axis=1).argsort()
            cols = dists.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                if dists[r, c] > TRACK_DISTANCE:
                    continue

                tid = track_ids[r]
                self.tracks[tid]['bbox'] = detections[c]['bbox']
                self.tracks[tid]['missing'] = 0

                # اگر قبلا ناشناس بود و الان شناخته شد
                if self.tracks[tid]['name'] == "Unknown" and detections[c]['name'] != "Unknown":
                    self.tracks[tid]['name'] = detections[c]['name']
                    self.tracks[tid]['role'] = detections[c]['role']
                    self.tracks[tid]['counted'] = False

                updated_tracks[tid] = self.tracks[tid]
                used_rows.add(r)
                used_cols.add(c)

            for i in range(len(new_centers)):
                if i not in used_cols:
                    self._add_track(updated_tracks, detections[i])

            for i, tid in enumerate(track_ids):
                if i not in used_rows:
                    self.tracks[tid]['missing'] += 1
                    if self.tracks[tid]['missing'] < MAX_MISSING_FRAMES:
                        updated_tracks[tid] = self.tracks[tid]

        elif new_centers:
            for det in detections:
                self._add_track(updated_tracks, det)
        else:
            for tid, trk in self.tracks.items():
                trk['missing'] += 1
                if trk['missing'] < MAX_MISSING_FRAMES:
                    updated_tracks[tid] = trk

        self.tracks = updated_tracks
        return self.tracks

    def _add_track(self, track_dict, det):
        track_dict[self.next_id] = {
            'bbox': det['bbox'],
            'missing': 0,
            'name': det['name'],
            'role': det['role'],
            'counted': False
        }
        self.next_id += 1

    def get_new_tracks(self):
        """لیست ترک‌های جدیدی که هنوز شمرده/ثبت نشده‌اند"""
        new_items = []
        for tid, trk in self.tracks.items():
            if not trk.get('counted', False):
                new_items.append((tid, trk))
                trk['counted'] = True
        return new_items


# ==========================================
#           کلاس اصلی سیستم
# ==========================================
class VanguardUniversitySystem:
    def __init__(self, root):
        self.root = root
        self.root.title("سامانه جامع کنترل تردد | VANGUARD AI")
        self.root.state('zoomed')

        self.style = ttk.Style("vapor")
        self.style.configure('.', font=('Vazir', 13))
        self.style.configure('Treeview', font=('Vazir', 12), rowheight=45)
        self.style.configure('Treeview.Heading', font=('Vazir', 14, 'bold'))

        self.db = DatabaseManager()
        self.tracker = FaceTracker()

        self.setup_paths()
        self.ensure_directories()

        self.is_running = False
        self.models_ready = False
        self.video_source = ""
        self.current_image_path = ""
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.recently_processed = {}
        self.PROCESSING_COOLDOWN = 5.0

        self.known_face_data = []

        # آمار روزانه
        self.count_registered_today = 0
        self.count_unknown_today = 0
        self.temp_reg_batch = 0
        self.temp_unk_batch = 0

        self._init_ui_variables()
        self.create_modern_ui()
        self.start_loading_models()
        self._start_db_timer()

    def _init_ui_variables(self):
        self.notebook = None
        self.tab_monitor = None
        self.tab_history = None
        self.tab_manage = None

        self.btn_start_cam = None
        self.btn_load_vid = None
        self.btn_load_img = None
        self.btn_stop = None

        self.lbl_fps = None
        self.lbl_status = None
        self.progress_bar = None

        self.live_view_frame = None
        self.live_lbl = None

        self.tree_inside = None

        # NEW: جدا کردن ماشین‌ها و افراد
        self.tree_live_vehicles = None
        self.tree_live_people = None

        self.tree_history = None
        self.lbl_preview = None

        self.entry_city = None
        self.entry_n3 = None
        self.combo_char = None
        self.entry_n2 = None
        self.entry_name = None
        self.combo_role = None
        self.entry_duration = None
        self.tree_perms = None

        self.model_plate = None
        self.model_char = None
        self.face_engine = None
        self.BLUR_THRESHOLD = 100

    def setup_paths(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.WEIGHTS_DIR = os.path.join(self.BASE_DIR, "weights")
        self.LOG_DIR = "logs"
        self.font_path = "C:/Windows/Fonts/tahoma.ttf"
        if not os.path.exists(self.font_path):
            self.font_path = "C:/Windows/Fonts/arial.ttf"

    def ensure_directories(self):
        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)

    def start_loading_models(self):
        threading.Thread(target=self._load_models_thread, daemon=True).start()

    def _load_models_thread(self):
        path_plate = os.path.join(self.WEIGHTS_DIR, "vanguard_plate_v2.pt")
        path_char = os.path.join(self.WEIGHTS_DIR, "vanguard_char_v3.pt")

        if not os.path.exists(path_plate):
            self.root.after(0, lambda: messagebox.showerror("Error", "فایل‌های مدل یافت نشد!"))
            return

        try:
            self.model_plate = YOLO(path_plate)
            self.model_char = YOLO(path_char)

            self.face_engine = FaceAnalysis(
                name='buffalo_s',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_engine.prepare(ctx_id=0, det_size=(640, 640))

            self.load_face_database()
            self.root.after(0, self._on_models_loaded)
        except Exception as e:
            print(f"Error loading models: {e}")

    def load_face_database(self):
        """
        لود کردن و دی‌کد کردن امبدینگ‌های چهره از دیتابیس
        """
        try:
            # فرض بر این است که get_all_faces سه لیست هم‌اندازه برمی‌گرداند
            encodings_list, names, roles = self.db.get_all_faces()

            self.known_face_data = []

            for i, raw_data in enumerate(encodings_list):
                vectors = []

                # --- مرحله ۱: تشخیص نوع داده و دی‌کد کردن ---
                if isinstance(raw_data, bytes):
                    # اگر داده به صورت باینری (BLOB) ذخیره شده باشد
                    try:
                        vectors = pickle.loads(raw_data)
                    except Exception:
                        print(f"❌ Error decoding pickle for {names[i]}")
                        continue
                elif isinstance(raw_data, str):
                    # اگر داده به صورت رشته متنی (JSON) ذخیره شده باشد
                    try:
                        vectors = json.loads(raw_data)
                    except Exception:
                        continue
                elif isinstance(raw_data, (list, np.ndarray)):
                    # اگر داده خام باشد
                    vectors = raw_data
                else:
                    continue

                # --- مرحله ۲: نرمال‌سازی بردارها ---
                # مطمئن می‌شویم که vectors لیستی از آرایه‌های نامپای است
                if isinstance(vectors, np.ndarray) and vectors.ndim == 1:
                    vectors = [vectors]

                norm_vectors = []
                for v in vectors:
                    v_np = np.array(v, dtype=np.float32)
                    norm = np.linalg.norm(v_np)
                    if norm > 0:
                        norm_vectors.append(v_np / norm)

                # --- مرحله ۳: افزودن به حافظه برنامه ---
                if norm_vectors:
                    self.known_face_data.append({
                        "name": names[i],
                        "role": roles[i],
                        "vectors": norm_vectors
                    })

            print(f"✅ {len(self.known_face_data)} Profiles Loaded Successfully.")

        except Exception as e:
            print(f"❌ Critical Error Loading Face Database: {e}")
            # برای جلوگیری از کرش کردن برنامه، لیست را خالی می‌گذاریم
            self.known_face_data = []

    def _on_models_loaded(self):
        self.models_ready = True
        self.lbl_status.config(text="سیستم آماده است ✅", bootstyle="success")
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_start_cam.config(state=NORMAL)
        self.btn_load_vid.config(state=NORMAL)
        self.btn_load_img.config(state=NORMAL)

    # --- تایمر ذخیره دیتابیس ---
    def _start_db_timer(self):
        if self.is_running:
            self.db.log_traffic_batch(self.temp_reg_batch, self.temp_unk_batch)
            self.temp_reg_batch = 0
            self.temp_unk_batch = 0
        self.root.after(30000, self._start_db_timer)

    # ==========================================
    #           منطق اصلی پردازش
    # ==========================================
    def process_frame(self, frame, is_video=False):
        # فریم خروجی که با cv2 روش نقاشی می‌کنیم (مشکل آبی هم با این طراحی حل می‌شود)
        img_cv = frame.copy()

        # ریسایز هوشمند برای AI (هم پلاک، هم چهره)
        h, w = frame.shape[:2]
        scale = 1.0
        if w > 1280:
            scale = 640 / w
            frame_ai = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            frame_ai = frame

        now = time.time()
        self._cleanup_cache(now)

        # --- A) تشخیص پلاک (روی frame_ai برای سرعت/پایداری) ---
        try:
            results = self.model_plate(frame_ai, verbose=False)
            for r in results:
                for box in r.boxes:
                    # اگر کلاس پلاک شما 30 است
                    if int(box.cls[0]) != 30:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # برگشت به مقیاس اصلی
                    if scale != 1.0:
                        x1 = int(x1 / scale); y1 = int(y1 / scale)
                        x2 = int(x2 / scale); y2 = int(y2 / scale)

                    # clamp
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w - 1, x2); y2 = min(h - 1, y2)

                    plate_id = f"plate_{x1}_{y1}"
                    if plate_id in self.recently_processed:
                        info, _ = self.recently_processed[plate_id]
                        img_cv = self._draw_overlay(img_cv, (x1, y1, x2, y2), info)
                        continue

                    plate_crop = frame[y1:y2, x1:x2]
                    raw_eng, main_fa, city_fa = self._detect_text_in_plate(plate_crop)

                    if raw_eng and self.validate_plate_format(raw_eng):
                        allowed, owner, role, _ = self.db.check_permission(raw_eng)
                        formatted = self.format_plate_display(raw_eng)
                        status_txt = "مجاز ✅" if allowed else "غیرمجاز ⛔"

                        # ذخیره لاگ + مسیر عکس (برای کلیک)
                        saved_path = self._log_and_save(
                            frame=frame,
                            raw=raw_eng,
                            ok=allowed,
                            owner=owner,
                            coords=(x1, y1, x2, y2),
                            is_face=False
                        )
                        self._update_vehicle_log(formatted, status_txt, saved_path)

                        info = (raw_eng, main_fa, city_fa, allowed)
                        self.recently_processed[plate_id] = (info, now)
                        img_cv = self._draw_overlay(img_cv, (x1, y1, x2, y2), info)
        except Exception as e:
            # بهتره silent نباشه؛ ولی فعلاً چاپ می‌کنیم
            # print(f"Plate Error: {e}")
            pass

        # --- B) تشخیص چهره و ردیابی ---
        try:
            faces = self.face_engine.get(frame_ai)
            detections = []

            for face in faces:
                box = (face.bbox / scale).astype(int)
                x1, y1, x2, y2 = map(int, box)
                # clamp
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w - 1, x2); y2 = min(h - 1, y2)

                name, role, score = self._identify_face(face.embedding)
                detections.append({'bbox': [x1, y1, x2, y2], 'name': name, 'role': role})

            tracks = self.tracker.update(detections)

            # ثبت افراد جدید (هر ترک یک بار)
            new_tracks = self.tracker.get_new_tracks()
            for tid, trk in new_tracks:
                box = trk['bbox']
                name = trk['name']
                role = trk['role']

                ok = (name != "Unknown")
                status_txt = "Registered ✅" if ok else "Unknown ⚠️"

                raw_id = name if ok else f"Unknown_{tid}"
                owner_for_db = role  # اینجا owner را نقش می‌گذاریم (یا می‌تونی name/role را ترکیب کنی)

                saved_path = self._log_and_save(
                    frame=frame,
                    raw=raw_id,
                    ok=ok,
                    owner=owner_for_db,
                    coords=(box[0], box[1], box[2], box[3]),
                    is_face=True
                )

                # آمار روزانه
                if ok:
                    self.count_registered_today += 1
                    self.temp_reg_batch += 1
                else:
                    self.count_unknown_today += 1
                    self.temp_unk_batch += 1

                # لاگ زنده افراد
                show_name = f"{name} ({role})" if ok else "Visitor"
                self._update_people_log(show_name, status_txt, saved_path)

            # رسم ترک‌ها
            for _, data in tracks.items():
                box = data['bbox']
                name = data['name']
                ok = (name != "Unknown")
                color = (0, 255, 0) if ok else (0, 165, 255)
                label = f"{name} ({data['role']})" if ok else "Visitor"

                cv2.rectangle(img_cv, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(img_cv, label, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # آمار روی تصویر
            stats = f"Registered: {self.count_registered_today} | Visitors: {self.count_unknown_today}"
            cv2.rectangle(img_cv, (0, 0), (520, 40), (0, 0, 0), -1)
            cv2.putText(img_cv, stats, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        except Exception as e:
            # print(f"Face Error: {e}")
            pass

        return img_cv, []

    def _identify_face(self, target_embedding):
        if not self.known_face_data:
            return "Unknown", "Visitor", 0.0

        # نرمال‌سازی چهره دیده شده در دوربین
        target_norm = target_embedding / (np.linalg.norm(target_embedding) + 1e-9)

        best_name = "Unknown"
        best_role = "Visitor"
        global_best_score = 0.0

        for person in self.known_face_data:
            # person['vectors'] لیستی از چند بردار (چند عکس مختلف از یک نفر) است
            # ما چهره فعلی را با تمام عکس‌های ذخیره شده آن فرد مقایسه می‌کنیم
            person_vectors = np.array(person["vectors"])  # تبدیل به ماتریس

            # ضرب نقطه‌ای (Cosine Similarity برای بردارهای نرمال شده)
            scores = np.dot(person_vectors, target_norm)

            # بیشترین شباهت بین عکس‌های این فرد را پیدا می‌کنیم
            max_score = float(np.max(scores))

            if max_score > global_best_score:
                global_best_score = max_score
                if global_best_score > FACE_THRESHOLD:
                    best_name = person["name"]
                    best_role = person["role"]

        return best_name, best_role, global_best_score

    # --- بقیه توابع کمکی ---
    @staticmethod
    def validate_plate_format(text: str) -> bool:
        if not text:
            return False
        clean = text.replace(" ", "").strip()
        return re.match(r'^[0-9]{2}[A-Za-z]+[0-9]{3}[0-9]{2}$', clean) is not None

    def convert_persian_input_to_english(self, p_num1, p_char, p_num2, p_city):
        try:
            en_num1 = "".join([PERSIAN_NUM_MAP.get(c, c) for c in p_num1])
            en_num2 = "".join([PERSIAN_NUM_MAP.get(c, c) for c in p_num2])
            en_city = "".join([PERSIAN_NUM_MAP.get(c, c) for c in p_city])
            en_char = PERSIAN_CHAR_MAP.get(p_char, p_char)
            return f"{en_num1}{en_char}{en_num2}{en_city}"
        except Exception:
            return None

    def format_plate_display(self, raw_plate: str) -> str:
        try:
            match = re.match(r'^([0-9]{2})([A-Za-z]+)([0-9]{3})([0-9]{2})$', raw_plate)
            if match:
                p1, ch, p2, city = match.groups()
                p1_fa = "".join([TRANSLATION_MAP.get(c, c) for c in p1])
                p2_fa = "".join([TRANSLATION_MAP.get(c, c) for c in p2])
                pc_fa = "".join([TRANSLATION_MAP.get(c, c) for c in city])
                ch_fa = TRANSLATION_MAP.get(ch, ch)
                return f"{pc_fa} ایران {p2_fa} {ch_fa} {p1_fa}"
            return raw_plate
        except Exception:
            return raw_plate

    def create_modern_ui(self):
        self.notebook = ttk.Notebook(self.root, bootstyle="primary")
        self.notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.tab_monitor = ttk.Frame(self.notebook)
        self.tab_history = ttk.Frame(self.notebook)
        self.tab_manage = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_monitor, text="  📡 مانیتورینگ  ")
        self.notebook.add(self.tab_history, text="  📂 سوابق  ")
        self.notebook.add(self.tab_manage, text="  🔐 مدیریت  ")

        self._setup_monitor_tab()
        self._setup_history_tab()
        self._setup_manage_tab()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    def _setup_monitor_tab(self):
        top_panel = ttk.Frame(self.tab_monitor, padding=10)
        top_panel.pack(fill=X)

        self.btn_start_cam = ttk.Button(
            top_panel, text="شروع دوربین 🔴", bootstyle="danger",
            state=DISABLED, command=self.start_webcam
        )
        self.btn_start_cam.pack(side=RIGHT, padx=5)

        self.btn_load_vid = ttk.Button(
            top_panel, text="فایل ویدئو 🎞️", bootstyle="warning",
            state=DISABLED, command=self.load_video
        )
        self.btn_load_vid.pack(side=RIGHT, padx=5)

        self.btn_load_img = ttk.Button(
            top_panel, text="عکس تکی 📷", bootstyle="info",
            state=DISABLED, command=self.load_single_image
        )
        self.btn_load_img.pack(side=RIGHT, padx=5)

        self.btn_stop = ttk.Button(top_panel, text="توقف", state=DISABLED, command=self.stop_stream)
        self.btn_stop.pack(side=RIGHT, padx=5)

        self.lbl_fps = ttk.Label(top_panel, text="FPS: 0", font=("Vazir", 12))
        self.lbl_fps.pack(side=LEFT)

        self.lbl_status = ttk.Label(top_panel, text="در حال بارگذاری هوش مصنوعی... ⏳", bootstyle="warning")
        self.lbl_status.pack(side=LEFT, padx=20)

        self.progress_bar = ttk.Progressbar(top_panel, mode='indeterminate', length=200, bootstyle="warning")
        self.progress_bar.pack(side=LEFT, padx=5)
        self.progress_bar.start(10)

        main_body = original_ttk.PanedWindow(self.tab_monitor, orient=HORIZONTAL)
        main_body.pack(fill=BOTH, expand=True, padx=5, pady=5)

        # تصویر زنده
        self.live_view_frame = ttk.Frame(main_body)
        self.live_lbl = ttk.Label(self.live_view_frame, text="Waiting for Signal...", anchor=CENTER)
        self.live_lbl.pack(fill=BOTH, expand=True)
        main_body.add(self.live_view_frame, weight=3)

        # ساید پنل
        side_panel = ttk.Frame(main_body)
        main_body.add(side_panel, weight=1)
        side_paned = original_ttk.PanedWindow(side_panel, orient=VERTICAL)
        side_paned.pack(fill=BOTH, expand=True)

        # خودروهای داخل
        frame_inside = ttk.Labelframe(side_paned, text="خودروهای داخل دانشگاه 🏫", padding=5)
        side_paned.add(frame_inside, weight=1)

        self.tree_inside = ttk.Treeview(
            frame_inside, columns=('plate', 'owner', 'time'), show='headings', bootstyle="success"
        )
        self.tree_inside.heading('plate', text='پلاک')
        self.tree_inside.column('plate', width=120, anchor=CENTER)
        self.tree_inside.heading('owner', text='مالک')
        self.tree_inside.column('owner', width=100, anchor=CENTER)
        self.tree_inside.heading('time', text='ورود')
        self.tree_inside.column('time', width=80, anchor=CENTER)
        self.tree_inside.pack(fill=BOTH, expand=True)

        ttk.Button(frame_inside, text="ثبت خروج دستی 🚗", bootstyle="outline-warning",
                   command=self.manual_exit).pack(fill=X, pady=5)

        # تردد لحظه‌ای (دو بخش: ماشین / افراد)
        frame_live = ttk.Labelframe(side_paned, text="ترددهای لحظه‌ای ⚡", padding=5)
        side_paned.add(frame_live, weight=1)

        live_paned = original_ttk.PanedWindow(frame_live, orient=VERTICAL)
        live_paned.pack(fill=BOTH, expand=True)

        # ماشین‌ها
        live_vehicle = ttk.Labelframe(live_paned, text="🚗 ماشین‌ها", padding=5)
        live_paned.add(live_vehicle, weight=1)
        self.tree_live_vehicles = ttk.Treeview(
            live_vehicle, columns=('plate', 'status', 'time', 'path'), show='headings', bootstyle="info"
        )
        self.tree_live_vehicles.heading('plate', text='پلاک')
        self.tree_live_vehicles.column('plate', width=140, anchor=CENTER)
        self.tree_live_vehicles.heading('status', text='وضعیت')
        self.tree_live_vehicles.column('status', width=90, anchor=CENTER)
        self.tree_live_vehicles.heading('time', text='زمان')
        self.tree_live_vehicles.column('time', width=80, anchor=CENTER)
        self.tree_live_vehicles.heading('path', text='Path')
        self.tree_live_vehicles.column('path', width=0, stretch=False)
        self.tree_live_vehicles.pack(fill=BOTH, expand=True)
        self.tree_live_vehicles.bind("<Double-1>", self.open_selected_vehicle_image)

        # افراد
        live_people = ttk.Labelframe(live_paned, text="🧑‍🤝‍🧑 افراد", padding=5)
        live_paned.add(live_people, weight=1)
        self.tree_live_people = ttk.Treeview(
            live_people, columns=('name', 'status', 'time', 'path'), show='headings', bootstyle="warning"
        )
        self.tree_live_people.heading('name', text='نام')
        self.tree_live_people.column('name', width=160, anchor=CENTER)
        self.tree_live_people.heading('status', text='وضعیت')
        self.tree_live_people.column('status', width=110, anchor=CENTER)
        self.tree_live_people.heading('time', text='زمان')
        self.tree_live_people.column('time', width=80, anchor=CENTER)
        self.tree_live_people.heading('path', text='Path')
        self.tree_live_people.column('path', width=0, stretch=False)
        self.tree_live_people.pack(fill=BOTH, expand=True)
        self.tree_live_people.bind("<Double-1>", self.open_selected_person_image)

    def _setup_history_tab(self):
        paned = original_ttk.PanedWindow(self.tab_history, orient=HORIZONTAL)
        paned.pack(fill=BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)

        toolbar = ttk.Frame(left_frame, padding=5)
        toolbar.pack(fill=X)
        ttk.Button(toolbar, text="بروزرسانی 🔄", command=self.refresh_history).pack(side=RIGHT)
        ttk.Button(toolbar, text="پاکسازی کل دیتابیس 🗑️", bootstyle="danger",
                   command=self.clear_database).pack(side=LEFT)

        self.tree_history = ttk.Treeview(
            left_frame,
            columns=('id', 'plate', 'status', 'time', 'path'),
            show='headings',
            bootstyle="info"
        )
        for col, text, width in [('id', 'ID', 50), ('plate', 'پلاک / نام', 200), ('status', 'وضعیت', 100),
                                 ('time', 'زمان', 150)]:
            self.tree_history.heading(col, text=text)
            self.tree_history.column(col, width=width, anchor=CENTER)

        self.tree_history.heading('path', text='Path')
        self.tree_history.column('path', width=0, stretch=False)

        self.tree_history.pack(fill=BOTH, expand=True)
        self.tree_history.bind("<<TreeviewSelect>>", self.show_preview_image)

        right_frame = ttk.Labelframe(paned, text="تصویر ثبت شده 📸", padding=10)
        paned.add(right_frame, weight=1)
        self.lbl_preview = ttk.Label(right_frame, text="انتخاب کنید", anchor=CENTER)
        self.lbl_preview.pack(fill=BOTH, expand=True)

    def _setup_manage_tab(self):
        frame_input = ttk.Labelframe(self.tab_manage, text="افزودن مجوز جدید", padding=20)
        frame_input.pack(fill=X, padx=20, pady=20)

        row1 = ttk.Frame(frame_input)
        row1.pack(fill=X, pady=5)

        ttk.Label(row1, text="پلاک:").pack(side=RIGHT, padx=5)
        self.entry_city = ttk.Entry(row1, width=3, justify=CENTER)
        self.entry_city.pack(side=RIGHT)
        self.entry_city.insert(0, "11")

        ttk.Label(row1, text="ایران").pack(side=RIGHT, padx=2)
        self.entry_n3 = ttk.Entry(row1, width=4, justify=CENTER)
        self.entry_n3.pack(side=RIGHT)

        self.combo_char = ttk.Combobox(row1, values=list(PERSIAN_CHAR_MAP.keys()), width=5, state="readonly")
        self.combo_char.pack(side=RIGHT, padx=2)
        self.combo_char.current(0)

        self.entry_n2 = ttk.Entry(row1, width=3, justify=CENTER)
        self.entry_n2.pack(side=RIGHT)

        row2 = ttk.Frame(frame_input)
        row2.pack(fill=X, pady=10)
        ttk.Label(row2, text="نام و نام خانوادگی:").pack(side=RIGHT, padx=5)
        self.entry_name = ttk.Entry(row2, width=30)
        self.entry_name.pack(side=RIGHT, padx=10)

        ttk.Label(row2, text="سمت:").pack(side=RIGHT, padx=5)
        self.combo_role = ttk.Combobox(row2, values=["دانشجو", "استاد", "کارمند", "مهمان"], state="readonly", width=10)
        self.combo_role.pack(side=RIGHT, padx=10)
        self.combo_role.current(0)

        row3 = ttk.Frame(frame_input)
        row3.pack(fill=X, pady=5)
        ttk.Label(row3, text="مدت اعتبار (برای مهمان - دقیقه):").pack(side=RIGHT, padx=5)
        self.entry_duration = ttk.Entry(row3, width=10)
        self.entry_duration.pack(side=RIGHT)
        self.entry_duration.insert(0, "0")
        ttk.Label(row3, text="(0 = دائم)").pack(side=RIGHT, padx=5)

        ttk.Button(frame_input, text="ثبت مجوز ✅", bootstyle="success", command=self.add_permission).pack(pady=10)

        frame_list = ttk.Labelframe(self.tab_manage, text="لیست پلاک‌های مجاز", padding=10)
        frame_list.pack(fill=BOTH, expand=True, padx=20, pady=10)

        self.tree_perms = ttk.Treeview(frame_list, columns=('raw', 'plate', 'owner', 'role', 'dur'), show='headings')
        for col, text, width in [('raw', 'ID', 0), ('plate', 'پلاک', 180), ('owner', 'نام', 100),
                                 ('role', 'سمت', 100), ('dur', 'زمان', 100)]:
            self.tree_perms.heading(col, text=text)
            self.tree_perms.column(col, width=width, anchor=CENTER, stretch=(width > 0))
        self.tree_perms.pack(fill=BOTH, expand=True)

        ttk.Button(frame_list, text="حذف انتخاب شده ❌", bootstyle="danger",
                   command=self.delete_permission).pack(pady=5)

    def add_permission(self):
        eng = self.convert_persian_input_to_english(
            self.entry_n2.get().strip(),
            self.combo_char.get().strip(),
            self.entry_n3.get().strip(),
            self.entry_city.get().strip()
        )
        if not eng or not self.validate_plate_format(eng):
            messagebox.showerror("خطا", "فرمت صحیح نیست")
            return

        try:
            d = int(self.entry_duration.get())
        except Exception:
            d = 0

        if self.db.add_permission(eng, self.entry_name.get(), self.combo_role.get(), d):
            messagebox.showinfo("موفق", "ثبت شد")
            self.refresh_permissions()
        else:
            messagebox.showerror("خطا", "تکراری")

    def delete_permission(self):
        s = self.tree_perms.selection()
        if s:
            r = self.tree_perms.item(s[0])['values'][0]
            if messagebox.askyesno("حذف", "حذف شود؟"):
                self.db.delete_permission(r)
                self.refresh_permissions()

    def refresh_permissions(self):
        for i in self.tree_perms.get_children():
            self.tree_perms.delete(i)
        for p in self.db.get_all_permissions():
            self.tree_perms.insert('', 'end', values=(p[0], self.format_plate_display(p[0]), p[1], p[2], p[3]))

    def show_preview_image(self, e=None):
        s = self.tree_history.selection()
        if s:
            p = self.tree_history.item(s[0])['values'][4]
            if os.path.exists(p):
                img = Image.open(p)
                img.thumbnail((300, 300))
                tk_img = ImageTk.PhotoImage(img)
                self.lbl_preview.config(image=tk_img, text="")
                self.lbl_preview.image = tk_img
            else:
                self.lbl_preview.config(image='', text="فایل نیست")

    def open_selected_vehicle_image(self, e=None):
        s = self.tree_live_vehicles.selection()
        if not s:
            return
        path = self.tree_live_vehicles.item(s[0])['values'][3]
        if not path or not os.path.exists(path):
            return

        win = ttk.Toplevel(self.root)
        win.title("تصویر پلاک")
        img = Image.open(path)
        img.thumbnail((900, 500))
        tk_img = ImageTk.PhotoImage(img)
        lbl = ttk.Label(win, image=tk_img)
        lbl.image = tk_img
        lbl.pack(padx=10, pady=10)

    def open_selected_person_image(self, e=None):
        s = self.tree_live_people.selection()
        if not s:
            return
        path = self.tree_live_people.item(s[0])['values'][3]
        if not path or not os.path.exists(path):
            return

        win = ttk.Toplevel(self.root)
        win.title("تصویر فرد")
        img = Image.open(path)
        img.thumbnail((900, 500))
        tk_img = ImageTk.PhotoImage(img)
        lbl = ttk.Label(win, image=tk_img)
        lbl.image = tk_img
        lbl.pack(padx=10, pady=10)

    def manual_exit(self):
        s = self.tree_inside.selection()
        if s:
            self.db.mark_exit(self.tree_inside.item(s[0])['values'][0])
            self.refresh_inside_list()

    def clear_database(self):
        if messagebox.askyesno("خطر", "کل اطلاعات پاک شود؟"):
            self.db.clear_database()
            self.refresh_history()
            self.refresh_inside_list()

    def refresh_history(self):
        for i in self.tree_history.get_children():
            self.tree_history.delete(i)
        for log in self.db.get_all_logs():
            self.tree_history.insert('', 'end',
                                     values=(log[0], self.format_plate_display(log[1]), log[2], log[3], log[4]))

    def refresh_inside_list(self):
        for i in self.tree_inside.get_children():
            self.tree_inside.delete(i)
        for v in self.db.get_vehicles_inside():
            self.tree_inside.insert('', 'end', values=(self.format_plate_display(v[0]), v[1], v[2]))

    def on_tab_change(self, e=None):
        t = self.notebook.tab(self.notebook.select(), "text").strip()
        if "سوابق" in t:
            self.refresh_history()
        if "مدیریت" in t:
            self.refresh_permissions()
        if "مانیتورینگ" in t:
            self.refresh_inside_list()

    def start_webcam(self):
        if not self.models_ready:
            return
        self.video_source = 0
        self._start_processing()

    def load_video(self):
        if not self.models_ready:
            return
        f = filedialog.askopenfilename(filetypes=[('Videos', '*.mp4 *.avi')])
        if f:
            self.video_source = f
            self._start_processing()

    def _start_processing(self):
        self.is_running = True
        self.btn_stop.config(state=NORMAL)
        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.ai_loop, daemon=True).start()
        self.update_ui_loop()

    def load_single_image(self):
        if not self.models_ready:
            return
        self.stop_stream()
        f = filedialog.askopenfilename(filetypes=[('Images', '*.jpg *.png')])
        if f:
            self.current_image_path = f
            try:
                fb = np.fromfile(f, dtype=np.uint8)
                img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                p, _ = self.process_frame(img, False)
                self.display_main_image(p)
            except Exception:
                pass

    def stop_stream(self):
        self.is_running = False
        self.btn_stop.config(state=DISABLED)

    def capture_loop(self):
        cap = None
        if isinstance(self.video_source, str) and not self.video_source.isdigit() and os.path.exists(self.video_source):
            cap = cv2.VideoCapture(self.video_source)
        else:
            for i in [0, 1, 2, 3]:
                t = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if t.isOpened() and t.read()[0]:
                    cap = t
                    cap.set(3, 1920)
                    cap.set(4, 1080)
                    break
                else:
                    t.release()

        if not cap:
            return

        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Exception:
                    pass
            self.frame_queue.put(frame)
            if isinstance(self.video_source, str) and not self.video_source.isdigit():
                time.sleep(0.01)
        cap.release()

    def ai_loop(self):
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except Exception:
                continue
            p, _ = self.process_frame(frame, True)
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except Exception:
                    pass
            self.result_queue.put(p)

    def _detect_text_in_plate(self, img):
        if img.size == 0:
            return None, "", ""
        z = cv2.resize(img, None, fx=2.5, fy=2.5)
        res = self.model_char(z, verbose=False, conf=0.55)
        chars = []
        for c in res[0].boxes:
            cls = int(c.cls[0])
            if cls < len(CLASS_NAMES):
                en = CLASS_NAMES[cls]
                fa = TRANSLATION_MAP.get(en, en)
                chars.append({'en': en, 'fa': fa, 'x': c.xyxy[0][0].item()})
        if not chars:
            return None, "", ""
        chars.sort(key=lambda x: x['x'])
        raw = "".join([c['en'] for c in chars])
        full = "".join([c['fa'] for c in chars])
        main = full[:-2] if len(full) >= 7 else full
        city = full[-2:] if len(full) >= 7 else "??"
        return raw, main, city

    # ✅ overlay اصلاح‌شده: cv2 برای bbox + تابع فارسی برای متن
    def _draw_overlay(self, cv_img, coords, info):
        x1, y1, x2, y2 = coords
        _, m, c, ok = info

        col = (0, 255, 0) if ok else (0, 0, 255)
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), col, 3)

        txt = f"{c} | {m}"
        ty = max(0, y1 - 35)
        cv2.rectangle(cv_img, (x1, ty), (x2, y1), (0, 0, 0), -1)
        return self.draw_persian_text(cv_img, txt, (x1 + 5, ty + 5), (255, 255, 255), 18)

    def draw_persian_text(self, cv_img, txt, pos, col, sz):
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype(self.font_path, sz)
        except Exception:
            f = ImageFont.load_default()
        bt = get_display(arabic_reshaper.reshape(txt))
        draw.text(pos, bt, font=f, fill=col)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # --- لاگ زنده: جدا ---
    def _update_vehicle_log(self, plate_text, status, path: Optional[str]):
        self.root.after(0, lambda: self._safe_vehicle_log(plate_text, status, path))

    def _safe_vehicle_log(self, plate_text, status, path: Optional[str]):
        if self.tree_live_vehicles:
            self.tree_live_vehicles.insert(
                '', 0,
                values=(plate_text, status, datetime.datetime.now().strftime("%H:%M:%S"), path or "")
            )
            if len(self.tree_live_vehicles.get_children()) > 60:
                self.tree_live_vehicles.delete(self.tree_live_vehicles.get_children()[-1])

    def _update_people_log(self, name_text, status, path: Optional[str]):
        self.root.after(0, lambda: self._safe_people_log(name_text, status, path))

    def _safe_people_log(self, name_text, status, path: Optional[str]):
        if self.tree_live_people:
            self.tree_live_people.insert(
                '', 0,
                values=(name_text, status, datetime.datetime.now().strftime("%H:%M:%S"), path or "")
            )
            if len(self.tree_live_people.get_children()) > 60:
                self.tree_live_people.delete(self.tree_live_people.get_children()[-1])

    # ✅ ذخیره و لاگ (برگشت مسیر فایل برای کلیک)
    def _log_and_save(self, frame, raw, ok, owner, coords, is_face=False) -> Optional[str]:
        try:
            fn = f"{'FACE' if is_face else 'CAR'}_{raw}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            fp = os.path.join(self.LOG_DIR, fn)

            status_db = "Allowed" if ok else "Denied"

            # فقط وقتی دیتابیس ثبت کرد، فایل هم ذخیره شود
            if self.db.log_entry(raw, status_db, fp, owner):
                x1, y1, x2, y2 = coords
                h, w = frame.shape[:2]
                c = frame[max(0, y1 - 50):min(h, y2 + 50), max(0, x1 - 50):min(w, x2 + 50)].copy()
                threading.Thread(target=cv2.imwrite, args=(fp, c), daemon=True).start()
                self.refresh_inside_list()
                return fp
        except Exception:
            pass
        return None

    def _cleanup_cache(self, now):
        for k in [k for k, v in self.recently_processed.items() if now - v[1] > self.PROCESSING_COOLDOWN]:
            del self.recently_processed[k]

    def update_ui_loop(self):
        if not self.is_running and self.result_queue.empty():
            return
        try:
            f = self.result_queue.get_nowait()
            self.display_main_image(f)
        except Exception:
            pass
        if self.is_running:
            self.root.after(10, self.update_ui_loop)

    def display_main_image(self, frame):
        w, h = self.live_view_frame.winfo_width(), self.live_view_frame.winfo_height()
        if w > 10:
            s = min(w / frame.shape[1], h / frame.shape[0])
            f = cv2.resize(frame, (0, 0), fx=s, fy=s)
            # ✅ اینجا RGB درست است و آبی نمی‌شود
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            self.live_lbl.config(image=img)
            self.live_lbl.image = img


if __name__ == "__main__":
    main_window = ttk.Window(themename="vapor")
    app = VanguardUniversitySystem(main_window)
    main_window.mainloop()
