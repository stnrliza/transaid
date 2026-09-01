import os
import cv2
import sqlite3
import threading
import numpy as np
import tkinter as tk
from tkinter import Canvas, Button, PhotoImage, messagebox
import customtkinter as ctk
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageTk
from start_push_button import PushButtonReader

# Path konstanta
ASSETS_PATH = Path(__file__).resolve().parent / "assets" / "frame-c"
PATIENTS_DATA_FOLDER = Path(__file__).resolve().parent / "Data_Pasien"
DATABASE_PATH = Path(__file__).resolve().parent / "pasien.db"

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class LiveCameraScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#FFFFFF")

        self.is_frozen = False
        self.cap = None
        self.camera_name = "Mencari Kamera..."
        self.last_frame = None
        self.is_active = False

        # Inisialisasi PushButtonReader secara opsional/aman
        try:
            self.push_button_reader = PushButtonReader(port='COM9', baudrate=115200, timeout=1)
        except Exception as e:
            print(f"PushButtonReader initialization skipped: {e}")
            self.push_button_reader = None

        # Variabel zoom, pan, dan mode tampilan
        self.zoom_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.mode = 'color'  # 'color' atau 'gray'

        # Variabel cropping
        self.crop_x = 620
        self.crop_y = 300
        self.crop_width = 320
        self.crop_height = 180

        # UI Container
        self.container = tk.Frame(self, bg="#FFFFFF")
        self.container.pack(fill="both", expand=True)

        # Header bar
        self.header_frame = tk.Frame(self.container, bg="#A8DFE6", height=90)
        self.header_frame.pack(fill="x", side="top")

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Pengambilan Gambar Kamera",
            font=("Poppins Bold", 26),
            text_color="#15218E"
        )
        self.title_label.place(relx=0.04, rely=0.5, anchor="w")

        # Info Status (Kamera & Serial)
        self.status_label = ctk.CTkLabel(
            self.header_frame,
            text="",
            font=("Poppins", 13),
            text_color="#16228E"
        )
        self.status_label.place(relx=0.55, rely=0.5, anchor="w")

        # Tombol Kembali di Header
        self.back_button = ctk.CTkButton(
            self.header_frame,
            text="Kembali",
            font=("Poppins Medium", 14),
            fg_color="#16228E",
            text_color="#FFFFFF",
            hover_color="#0e1761",
            corner_radius=10,
            width=100,
            height=36,
            command=self.go_back
        )
        self.back_button.place(relx=0.94, rely=0.5, anchor="e")

        # Main video area canvas
        self.canvas = Canvas(self.container, bg="#1E1E1E", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=20, pady=(10, 10))
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw")

        # Overlay freeze text
        self.freeze_text_id = self.canvas.create_text(
            30, 30, anchor="nw",
            text="",
            fill="#E74C3C",
            font=("Poppins Bold", 20)
        )

        # Bottom Control Bar
        self.control_frame = tk.Frame(self.container, bg="#FFFFFF", height=80)
        self.control_frame.pack(fill="x", side="bottom", padx=20, pady=(0, 15))

        # Tombol restart/unfreeze live feed
        restart_icon_path = relative_to_assets("c-button-2.png")
        if restart_icon_path.exists():
            self.restart_img = PhotoImage(file=restart_icon_path)
            self.restart_btn = Button(
                self.control_frame,
                image=self.restart_img,
                borderwidth=0,
                highlightthickness=0,
                command=self.reset_live_feed,
                relief="flat",
                background="#FFFFFF",
                activebackground="#F0F4F8"
            )
            self.restart_btn.pack(side="left", padx=10)
        else:
            self.restart_btn = ctk.CTkButton(
                self.control_frame,
                text="Ulangi Kamera",
                font=("Poppins Medium", 14),
                fg_color="#7F8C8D",
                text_color="#FFFFFF",
                hover_color="#636e72",
                corner_radius=12,
                width=140,
                height=45,
                command=self.reset_live_feed
            )
            self.restart_btn.pack(side="left", padx=10)

        # Zoom & Pan Info Hint
        self.hint_label = ctk.CTkLabel(
            self.control_frame,
            text="Kontrol: [I] Zoom In | [O] Zoom Out | [W/A/S/D] Geser | [C] Warna | [G] Grayscale",
            font=("Poppins", 12),
            text_color="#555555"
        )
        self.hint_label.pack(side="left", padx=20)

        # Tombol Tangkap & Lanjutkan ("Selesai")
        self.finish_button = ctk.CTkButton(
            self.control_frame,
            text="Selesai & Diagnosis >>",
            font=("Poppins Bold", 16),
            fg_color="#2ECC71",
            text_color="#FFFFFF",
            hover_color="#27ae60",
            corner_radius=12,
            width=220,
            height=48,
            command=self.on_finish_clicked
        )
        self.finish_button.pack(side="right", padx=10)

        # Inisialisasi kamera & status
        self.init_camera_flexible()
        self.update_status_display()

        # Binding keyboard untuk zoom dan manipulasi gambar
        self.bind_keys()

        # Monitoring push button di background thread
        threading.Thread(target=self.monitor_push_button, daemon=True).start()

    def bind_keys(self):
        """Binding keyboard shortcut untuk navigasi dan penyesuaian kamera."""
        self.controller.bind("<KeyPress-i>", lambda e: self.adjust_zoom(0.2))
        self.controller.bind("<KeyPress-o>", lambda e: self.adjust_zoom(-0.2))
        self.controller.bind("<KeyPress-w>", lambda e: self.adjust_pan(0, -30))
        self.controller.bind("<KeyPress-s>", lambda e: self.adjust_pan(0, 30))
        self.controller.bind("<KeyPress-a>", lambda e: self.adjust_pan(-30, 0))
        self.controller.bind("<KeyPress-d>", lambda e: self.adjust_pan(30, 0))
        self.controller.bind("<KeyPress-c>", lambda e: self.set_mode('color'))
        self.controller.bind("<KeyPress-g>", lambda e: self.set_mode('gray'))

    def init_camera_flexible(self):
        """
        Mencoba membuka kamera dengan fleksibel:
        1. Coba Kamera Eksternal (Indeks 1 dengan DSHOW)
        2. Jika gagal, coba Kamera Internal (Indeks 0 dengan DSHOW)
        3. Jika gagal, coba Kamera Default (Indeks 0 standard)
        """
        if self.cap is not None and self.cap.isOpened():
            return

        # 1. Coba Kamera Eksternal (indeks 1)
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap = cap
                self.camera_name = "Kamera Eksternal (Indeks 1)"
                print(f"[Camera] {self.camera_name} berhasil dibuka.")
                return
            cap.release()

        # 2. Coba Kamera Internal (indeks 0 dengan DSHOW)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap = cap
                self.camera_name = "Kamera Internal (Indeks 0)"
                print(f"[Camera] {self.camera_name} berhasil dibuka.")
                return
            cap.release()

        # 3. Fallback standard tanpa DSHOW
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            self.cap = cap
            self.camera_name = "Kamera Standard (Indeks 0)"
            print(f"[Camera] {self.camera_name} berhasil dibuka.")
            return

        # Tidak ada kamera terdeteksi
        self.cap = None
        self.camera_name = "Kamera Tidak Terdeteksi"
        print("[Camera] Peringatan: Tidak ada kamera yang terdeteksi.")

    def update_status_display(self):
        """Memperbarui teks status kamera dan push button di header."""
        is_serial_active = (
            self.push_button_reader is not None and 
            self.push_button_reader.is_connected
        )
        serial_text = "🟢 Hardware PushButton: Terhubung" if is_serial_active else "⚪ PushButton: Nonaktif (Gunakan Tombol Layar)"
        cam_text = f"📷 {self.camera_name}"

        patient_name = getattr(self.controller, 'current_patient_nama', '')
        p_info = f" | Pasien: {patient_name}" if patient_name else ""

        self.status_label.configure(text=f"{cam_text} | {serial_text}{p_info}")

    def on_show(self):
        """Dipanggil saat frame ini ditampilkan ke layar."""
        self.is_active = True
        self.is_frozen = False
        self.canvas.itemconfig(self.freeze_text_id, text="")
        
        # Pastikan kamera menyala
        if self.cap is None or not self.cap.isOpened():
            self.init_camera_flexible()
            
        self.update_status_display()
        self.update_frame()

    def on_hide(self):
        """Dipanggil ketika beralih dari frame ini."""
        self.is_active = False

    def go_back(self):
        """Kembali ke layar data pasien."""
        self.is_active = False
        self.controller.show_frame("PatientDataScreen")

    def update_frame(self):
        """Mengambil frame dari kamera dan memperbarui tampilan canvas."""
        if not self.is_active:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1:
            cw = 1280
        if ch <= 1:
            ch = 720

        if not self.is_frozen and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame.copy()
                display_frame = self.process_frame_display(frame, cw, ch)
                img = Image.fromarray(display_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.itemconfig(self.image_on_canvas, image=imgtk)
                self.canvas.coords(self.image_on_canvas, cw // 2, ch // 2)
                self.canvas.itemconfig(self.image_on_canvas, anchor="center")
                self.canvas.image = imgtk
            else:
                self.render_synthetic_frame(cw, ch, "Gagal Membaca Frame Kamera")
        elif not self.cap or not self.cap.isOpened():
            self.render_synthetic_frame(cw, ch, "Kamera Tidak Terdeteksi\n(Klik 'Selesai & Diagnosis' untuk Simulasi)")

        # Loop pembaruan frame setiap 30 ms (~33 FPS)
        if self.is_active:
            self.after(30, self.update_frame)

    def process_frame_display(self, frame, canvas_w, canvas_h):
        """Menerapkan zoom, pan, mode warna/grayscale, dan resize ke canvas."""
        h, w = frame.shape[:2]

        # Terapkan zoom
        if self.zoom_scale > 1.0:
            new_w = int(w * self.zoom_scale)
            new_h = int(h * self.zoom_scale)
            frame_resized = cv2.resize(frame, (new_w, new_h))
            
            start_x = int(np.clip(self.offset_x, 0, max(0, new_w - w)))
            start_y = int(np.clip(self.offset_y, 0, max(0, new_h - h)))
            frame = frame_resized[start_y:start_y + h, start_x:start_x + w]

        # Mode warna / grayscale
        if self.mode == 'gray':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Scale agar fit di dalam canvas
        fh, fw = frame.shape[:2]
        scale = min(canvas_w / fw, canvas_h / fh)
        target_w = max(1, int(fw * scale))
        target_h = max(1, int(fh * scale))
        
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def render_synthetic_frame(self, w, h, text):
        """Membuat canvas placeholder elegan jika kamera fisik tidak terpasang."""
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        dummy[:] = (30, 30, 35)  # Dark slate background

        # Gambar kotak viewfinder simulasi
        cv2.rectangle(dummy, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (168, 223, 230), 2)

        # Simpan frame dummy sebagai last_frame agar proses tetap dapat dilanjutkan
        if self.last_frame is None:
            synth = np.full((720, 1280, 3), 120, dtype=np.uint8)
            cv2.putText(synth, "TransAID Dental Capture Simulation", (200, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            self.last_frame = synth

        lines = text.split('\n')
        for idx, line in enumerate(lines):
            y_pos = h // 2 - 20 + (idx * 40)
            (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x_pos = max(20, (w - tw) // 2)
            cv2.putText(dummy, line, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        img = Image.fromarray(dummy)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.itemconfig(self.image_on_canvas, image=imgtk)
        self.canvas.coords(self.image_on_canvas, w // 2, h // 2)
        self.canvas.itemconfig(self.image_on_canvas, anchor="center")
        self.canvas.image = imgtk

    def adjust_zoom(self, delta):
        self.zoom_scale = np.clip(round(self.zoom_scale + delta, 2), 1.0, 4.5)
        print(f"Zoom level: {self.zoom_scale}x")

    def adjust_pan(self, dx, dy):
        self.offset_x = max(0, self.offset_x + dx)
        self.offset_y = max(0, self.offset_y + dy)

    def set_mode(self, mode):
        self.mode = mode
        print(f"Camera mode: {mode}")

    def freeze_frame(self):
        """Membekukan live feed saat tombol push button ditekan."""
        if not self.is_frozen:
            self.is_frozen = True
            self.canvas.itemconfig(self.freeze_text_id, text="[FROZEN - Gambar Terkunci]")
            print("[Camera] Live feed dibekukan.")

    def reset_live_feed(self):
        """Memulai kembali live feed jika sebelumnya di-freeze."""
        if self.is_frozen:
            self.is_frozen = False
            self.canvas.itemconfig(self.freeze_text_id, text="")
            print("[Camera] Live feed dimulai ulang.")

    def on_finish_clicked(self):
        """
        Handler saat tombol 'Selesai & Diagnosis' ditekan.
        - Jika serial TIDAK terhubung: tombol ini otomatis membekukan/mengambil frame aktif dan lanjut.
        - Jika serial terhubung: menggunakan frame yang sudah di-freeze (atau membekukan jika belum) lalu lanjut.
        """
        is_serial_connected = (
            self.push_button_reader is not None and 
            self.push_button_reader.is_connected
        )

        if not is_serial_connected:
            # Mode fleksibel tanpa hardware push button: freeze otomatis & proses
            self.is_frozen = True
        else:
            # Mode dengan serial push button
            if not self.is_frozen:
                self.is_frozen = True

        self.save_capture_and_proceed()

    def save_capture_and_proceed(self):
        """
        Menyimpan gambar hasil tangkapan (original, cropped, grayscale)
        ke folder pasien aktif dan melanjutkan ke LoadingScreen.
        """
        # Tentukan folder penyimpanan
        folder_path = getattr(self.controller, 'current_patient_folder', None)
        if not folder_path:
            tgl = datetime.now().strftime("%Y-%m-%d")
            folder_path = PATIENTS_DATA_FOLDER / f"{tgl}_Pasien-Manual"
            folder_path.mkdir(parents=True, exist_ok=True)
            self.controller.current_patient_folder = folder_path

        # Dapatkan frame yang akan disimpan
        frame_to_save = None
        if self.last_frame is not None:
            frame_to_save = self.last_frame.copy()
        elif self.cap and self.cap.isOpened():
            ret, f = self.cap.read()
            if ret:
                frame_to_save = f

        if frame_to_save is None:
            # Buat fallback image jika tidak ada frame sama sekali
            frame_to_save = np.full((720, 1280, 3), 150, dtype=np.uint8)
            cv2.putText(frame_to_save, "Sample Dental Image", (300, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 1. Simpan gambar asli
        original_img_path = Path(folder_path) / "captured-image-original.png"
        cv2.imwrite(str(original_img_path), frame_to_save)
        print(f"Gambar asli disimpan: {original_img_path}")

        # 2. Crop gambar
        cropped_frame = self.apply_crop(frame_to_save)
        cropped_img_path = Path(folder_path) / "captured-image-cropped.png"
        cv2.imwrite(str(cropped_img_path), cropped_frame)
        print(f"Gambar crop disimpan: {cropped_img_path}")

        # 3. Simpan versi grayscale cropped
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        bw_cropped_path = Path(folder_path) / "captured-image-bw-cropped.png"
        cv2.imwrite(str(bw_cropped_path), gray_frame_bgr)
        print(f"Gambar grayscale crop disimpan: {bw_cropped_path}")

        # 4. Simpan path gambar asli ke database pasien
        patient_id = getattr(self.controller, 'current_patient_id', None)
        if patient_id and DATABASE_PATH.exists():
            try:
                conn = sqlite3.connect(DATABASE_PATH)
                c = conn.cursor()
                c.execute("UPDATE pasien SET path_gambar = ? WHERE id = ?", (str(original_img_path), patient_id))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Error updating path_gambar in database: {e}")

        # 5. Nonaktifkan kamera saat beralih ke loading
        self.is_active = False

        # 6. Jalankan segmentasi dan beralih ke LoadingScreen
        if "LoadingScreen" in self.controller.frames:
            loading_screen = self.controller.frames["LoadingScreen"]
            self.controller.show_frame("LoadingScreen")
            loading_screen.start_segmentation(str(bw_cropped_path), str(folder_path))

    def apply_crop(self, frame):
        """Crop frame sesuai area ROI (Region of Interest)."""
        h, w = frame.shape[:2]
        sx = min(self.crop_x, max(0, w - self.crop_width))
        sy = min(self.crop_y, max(0, h - self.crop_height))
        ex = min(w, sx + self.crop_width)
        ey = min(h, sy + self.crop_height)
        return frame[sy:ey, sx:ex]

    def monitor_push_button(self):
        """Thread background untuk memonitor hardware push button."""
        while True:
            if self.push_button_reader and self.push_button_reader.is_connected:
                status = self.push_button_reader.read_push_button_status()
                if status == 'short_press' and not self.is_frozen:
                    self.after(0, self.freeze_frame)
            threading.Event().wait(0.1)
