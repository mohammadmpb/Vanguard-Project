import sys
import os
import time
import threading
import traceback
import tkinter as tk
from tkinter import ttk, messagebox



# --- سیستم لاگ‌برداری (غیرفعال برای نسخه نهایی) ---
# --- سیستم لاگ‌برداری (فعال سازی مجدد برای دیباگ) ---
import datetime
def log(msg):
    # چاپ کردن پیام به همراه ساعت دقیق
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush() # مطمئن شو که همون لحظه چاپ میشه


log("--- STARTUP INITIATED ---")

# تنظیمات حیاتی برای ویندوز 7
os.environ["OMP_NUM_THREADS"] = "1"
log("Environment variable OMP_NUM_THREADS set to 1")

try:
    log("1. Importing PIL...")
    from PIL import Image, ImageTk

    log("2. Importing OpenCV...")
    import cv2

    log(f"   OpenCV Version: {cv2.__version__}")

    log("3. Importing Numpy...")
    import numpy as np

    log("4. Importing Database...")
    from database import DatabaseManager

    log("5. Importing InsightFace (This is heavy)...")
    from insightface.app import FaceAnalysis

except ImportError as e:
    log(f"CRITICAL IMPORT ERROR: {e}")
    input("Press Enter to exit...")
    sys.exit(1)
except Exception as e:
    log(f"UNKNOWN STARTUP ERROR: {e}")
    input("Press Enter to exit...")
    sys.exit(1)

log("--- IMPORTS COMPLETE ---")


class EnrollmentApp:
    def __init__(self, root):
        log("Initializing UI...")
        self.root = root
        self.root.title("ثبت هویت نظامی (DEBUG MODE) | VANGUARD")
        self.root.geometry("1050x700")

        self.style = ttk.Style()
        self.style.theme_use('clam')

        # اتصال به دیتابیس
        try:
            log("Connecting to Database...")
            self.db = DatabaseManager()
            log("Database Connected.")
        except Exception as e:
            log(f"Database Error: {e}")
            messagebox.showerror("خطای دیتابیس", f"فایل دیتابیس یا کلید امنیتی یافت نشد!\n{e}")
            sys.exit(1)

        self.app = None
        self.cap = None
        self.is_running = True
        self.collected_embeddings = []

        # لیست مراحل
        self.steps = [
            {'id': 'center', 'msg': '1. مستقیم نگاه کنید (Center)', 'min': -10, 'max': 10},
            {'id': 'right', 'msg': '2. آرام به راست (Right) ➡', 'min': 15, 'max': 50},
            {'id': 'left', 'msg': '3. آرام به چپ (Left) ⬅', 'min': -50, 'max': -15}
        ]
        self.current_step_index = 0
        self.known_face_encodings = []
        self.known_face_names = []

        # شروع تردها
        threading.Thread(target=self.load_face_database, daemon=True).start()
        self.setup_ui()
        threading.Thread(target=self._load_resources, daemon=True).start()

    def load_face_database(self):
        try:
            log("Loading existing faces from DB...")
            encodings_list, names, _ = self.db.get_all_faces()
            self.known_face_encodings = []
            self.known_face_names = names

            for data_blob in encodings_list:
                if isinstance(data_blob, np.ndarray):
                    vectors = [data_blob]
                elif isinstance(data_blob, list):
                    vectors = data_blob
                else:
                    continue
                norm_vectors = [v / np.linalg.norm(v) for v in vectors]
                self.known_face_encodings.append(norm_vectors)
            log(f"Loaded {len(names)} users from DB.")
        except Exception as e:
            log(f"DB Load Error: {e}")

    def check_duplicate_face(self, new_embedding):
        if not self.known_face_encodings:
            return None
        target_norm = new_embedding / np.linalg.norm(new_embedding)
        for i, person_vectors in enumerate(self.known_face_encodings):
            for p_vec in person_vectors:
                score = np.dot(p_vec, target_norm)
                if score > 0.60:
                    return self.known_face_names[i]
        return None

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.left_panel = ttk.LabelFrame(main_frame, text="اطلاعات پرسنلی", padding=15)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Label(self.left_panel, text="نام و نام خانوادگی:").pack(fill=tk.X, pady=(10, 0))
        self.entry_name = ttk.Entry(self.left_panel, state='disabled')
        self.entry_name.pack(fill=tk.X, pady=5)

        ttk.Label(self.left_panel, text="کد ملی (ID):").pack(fill=tk.X, pady=(10, 0))
        self.entry_id = ttk.Entry(self.left_panel, state='disabled')
        self.entry_id.pack(fill=tk.X, pady=5)

        ttk.Label(self.left_panel, text="سمت (Role):").pack(fill=tk.X, pady=(10, 0))
        roles = ["Professor", "Student", "Dorm_Resident", "Staff"]
        self.combo_role = ttk.Combobox(self.left_panel, values=roles, state='disabled')
        self.combo_role.pack(fill=tk.X, pady=5)
        self.combo_role.current(1)

        ttk.Separator(self.left_panel).pack(fill=tk.X, pady=20)

        self.lbl_status = ttk.Label(self.left_panel, text="در حال لود مدل‌ها (صبر کنید)...", foreground="red",
                                    wraplength=200, justify="center")
        self.lbl_status.pack(pady=10)

        self.progress = ttk.Progressbar(self.left_panel, value=0, maximum=3)
        self.progress.pack(fill=tk.X, pady=10)

        self.btn_save = ttk.Button(self.left_panel, text="ذخیره و رمزنگاری 🔒", state='disabled', command=self.save_user)
        self.btn_save.pack(fill=tk.X, pady=10)

        self.btn_reset = ttk.Button(self.left_panel, text="شروع مجدد 🔄", state='disabled', command=self.reset_steps)
        self.btn_reset.pack(fill=tk.X)

        right_panel = ttk.LabelFrame(main_frame, text="تصویر زنده", padding=5)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.lbl_camera = ttk.Label(right_panel)
        self.lbl_camera.pack(fill=tk.BOTH, expand=True)

    def _load_resources(self):
        try:
            log("--- RESOURCE LOADER STARTED ---")

            # 1. Check Model Path
            user_home = os.path.expanduser('~')
            model_path = os.path.join(user_home, '.insightface')
            log(f"Checking for models in: {model_path}")

            if os.path.exists(model_path):
                log("✅ Model folder FOUND.")
                files = os.listdir(model_path)
                log(f"   Contents: {files}")
            else:
                log("❌ WARNING: Model folder NOT FOUND! Application might hang trying to download.")

            # 2. Load InsightFace
            log("Instantiating FaceAnalysis...")
            self.app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

            log("Preparing Models (det_size=640)...")
            start_time = time.time()
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            log(f"✅ Models Prepared in {time.time() - start_time:.2f} seconds.")

            # 3. Camera Check
            log("Searching for Cameras...")
            found_cap = None
            for port in [0, 1, 2]:
                log(f"   Testing Port {port}...")
                temp_cap = cv2.VideoCapture(port, cv2.CAP_DSHOW)  # DSHOW for Windows
                if temp_cap.isOpened():
                    ret, frame = temp_cap.read()
                    if ret:
                        log(f"   ✅ Camera FOUND on Port {port}")
                        found_cap = temp_cap
                        # تنظیمات رزولوشن (موقتا غیرفعال برای جلوگیری از صفحه سیاه)
                        # found_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        # found_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        break
                    else:
                        log(f"   ❌ Port {port} opened but returned no frame.")
                        temp_cap.release()
                else:
                    log(f"   ❌ Port {port} failed to open.")

            if found_cap is None:
                log("❌ FATAL: No working camera found.")
                self.root.after(0, lambda: messagebox.showerror("Error", "No Camera Found!"))
                return

            self.cap = found_cap
            self.root.after(0, self._on_ready)

        except Exception as e:
            log(f"❌ CRITICAL ERROR IN LOADER: {e}")
            traceback.print_exc()

    def _on_ready(self):
        log("System Ready. UI Enabled.")
        self.entry_name.config(state='normal')
        self.entry_id.config(state='normal')
        self.combo_role.config(state='readonly')
        self.btn_reset.config(state='normal')
        self.update_instruction()
        self.process_webcam()

    def get_yaw(self, face):
        return face.pose[1] if face.pose is not None else 0

    def update_instruction(self):
        if self.current_step_index < len(self.steps):
            step = self.steps[self.current_step_index]
            self.lbl_status.config(text=step['msg'], foreground="blue")
        else:
            self.lbl_status.config(text="✅ اسکن تکمیل شد. دکمه ذخیره فعال است.", foreground="green")
            self.btn_save.config(state='normal')

    def process_webcam(self):
        if not self.is_running: return

        try:
            ret, frame = self.cap.read()
            if not ret:
                self.lbl_status.config(text="❌ دوربین قطع شد!", foreground="red")
            else:
                display_frame = cv2.resize(frame, (640, 480))
                display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                faces = self.app.get(display_frame)

                if len(faces) == 1:
                    face = faces[0]
                    box = face.bbox.astype(int)
                    yaw = self.get_yaw(face)
                    cv2.rectangle(display_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                    if self.current_step_index < 3:
                        if self.current_step_index == 0:
                            duplicate_name = self.check_duplicate_face(face.embedding)
                            if duplicate_name:
                                self.lbl_status.config(text=f"⛔ اخطار: {duplicate_name} قبلا ثبت شده!",
                                                       foreground="red")
                                cv2.rectangle(display_rgb, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                                self._update_ui_image(display_rgb)
                                self.root.after(10, self.process_webcam)
                                return

                        step = self.steps[self.current_step_index]
                        if step['min'] <= yaw <= step['max']:
                            if face.det_score > 0.60:
                                self.collected_embeddings.append(face.embedding)
                                self.current_step_index += 1
                                self.progress['value'] = self.current_step_index
                                self.update_instruction()
                                cv2.rectangle(display_rgb, (0, 0), (640, 480), (0, 255, 0), 10)

                elif len(faces) > 1:
                    self.lbl_status.config(text="⚠️ فقط یک نفر!", foreground="orange")

                self._update_ui_image(display_rgb)

        except Exception as e:
            print(f"Frame Error: {e}")

        self.root.after(30, self.process_webcam)

    def _update_ui_image(self, frame_rgb):
        try:
            img_pil = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.lbl_camera.imgtk = imgtk
            self.lbl_camera.configure(image=imgtk)
        except:
            pass

    def save_user(self):
        name = self.entry_name.get().strip()
        nid = self.entry_id.get().strip()
        role = self.combo_role.get()

        if not name or not nid.isdigit():
            messagebox.showerror("خطا", "لطفا نام و کد ملی صحیح وارد کنید.")
            return

        if len(self.collected_embeddings) != 3:
            messagebox.showerror("خطا", "اسکن کامل نیست.")
            return

        if self.db.add_face_user(nid, name, role, self.collected_embeddings):
            messagebox.showinfo("موفقیت", f"اطلاعات {name} با موفقیت رمزنگاری و ثبت شد.")
            self.load_face_database()
            self.reset_steps()
            self.entry_name.delete(0, tk.END)
            self.entry_id.delete(0, tk.END)
        else:
            messagebox.showerror("خطا", "خطا در دیتابیس (احتمالاً کد ملی تکراری).")

    def reset_steps(self):
        self.collected_embeddings = []
        self.current_step_index = 0
        self.progress['value'] = 0
        self.btn_save.config(state='disabled')
        self.update_instruction()

    def on_close(self):
        self.is_running = False
        if self.cap: self.cap.release()
        log("App Closing...")
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    try:
        log("Creating Root Window...")
        root = tk.Tk()
        app = EnrollmentApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        log("Mainloop Started.")
        root.mainloop()
    except Exception as e:
        log(f"CRITICAL MAIN ERROR: {e}")
        input("Press Enter to exit...")