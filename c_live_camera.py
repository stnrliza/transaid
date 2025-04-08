import ctypes
import customtkinter as ctk
import cv2
import os
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageTk
from tkinter import Canvas, Button, PhotoImage
from start_push_button import PushButtonReader
from b1_patient_data import PATIENTS_DATA_FOLDER

# Path untuk aset
ASSETS_PATH = Path(__file__).resolve().parent / "assets" / "frame-c"

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Mengaktifkan DPI Awareness
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception as e:
    print(f"Error setting DPI awareness: {e}")

class LiveCameraScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#FFFFFF")
        self.is_frozen = False
        self.push_button_reader = PushButtonReader(port='COM9', baudrate=115200, timeout=1)

        tanggal_pemeriksaan = datetime.now().strftime("%Y-%m-%d")
        patient_number = LiveCameraScreen.get_last_patient_number() + 1
        folder_name = PATIENTS_DATA_FOLDER / f"{tanggal_pemeriksaan}_Pasien-{patient_number}"
        folder_name.mkdir(parents=True, exist_ok=True)
        print(f"Folder created: {folder_name}")

        # Variabel untuk zoom, posisi, dan mode warna/hitam-putih
        self.zoom_scale = 4.5  # Mulai dengan zoom 4.5x
        self.offset_x = 0
        self.offset_y = 0
        self.mode = 'color'  # Mode default warna

        # Menetapkan nilai manual offset sesuai koordinat yang diperoleh
        self.manual_offset_x = 2820.0  # Sesuai dengan koordinat yang diperoleh
        self.manual_offset_y = 1560.0  # Sesuai dengan koordinat yang diperoleh

        # Variabel cropping
        self.crop_x = 620
        self.crop_y = 300
        self.crop_width = 320
        self.crop_height = 180

        # Membuat canvas
        self.canvas = Canvas(self, bg="#FFFFFF", height=1080, width=1920)
        self.canvas.pack(fill="both", expand=True)

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        self.canvas.create_rectangle(
            0.0,
            0.0,
            1920.0,
            104.22857666015625,
            fill="#A8DFE6",
            outline=""
        )

        self.canvas.create_text(
            85,
            20,
            anchor="nw",
            text="Menangkap Gambar",
            fill="#15218E",
            font=("Poppins Bold", 43 * -1)
        )

        # Placeholder untuk frame video
        self.image_on_canvas = self.canvas.create_image(960, 545, anchor="center")

        # Inisialisasi kamera eksternal (gunakan indeks 0)
        self.cap = self.init_external_camera()

        # Tombol simpan dan navigasi ke LoadingScreen
        self.button_font_size = int(screen_height / 54)
        finish_button = ctk.CTkButton(
            self.canvas,
            text="Selesai",
            font=("Poppins Medium", self.button_font_size),
            fg_color="#A8DEE6",
            text_color="#16228E",
            corner_radius=15,
            command=lambda: [self.finish_capture(force_freeze=True), self.controller.show_frame("LoadingScreen")]
        )
        finish_button.place(relx=0.8, rely=0.8, anchor="nw", relwidth=0.2, relheight=0.1)

        # Tombol "Restart Live Feed"
        button_image_2 = PhotoImage(file=relative_to_assets("c-button-2.png"))
        button_2 = Button(
            self,
            image=button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.reset_live_feed(),
            relief="flat",
            background="#ffffff"
        )
        button_2.place(x=86, y=791, width=150, height=150)  # Diubah posisinya
        self.button_image_2 = button_image_2

        # Mulai menangkap video
        self.update_frame()

        # Mulai monitoring push button di thread terpisah
        threading.Thread(target=self.monitor_push_button, daemon=True).start()

        # Tambahkan binding key untuk zoom, geser, dan mode hitam-putih/warna
        self.controller.bind("<KeyPress>", self.keypress_event)

    def init_external_camera(self):
        """
        Membuka kamera eksternal (indeks 0) sebagai default.
        """
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            print("Kamera eksternal terbuka dengan indeks 0.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set resolusi kamera
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Gunakan manual offset yang sudah ditetapkan
            self.offset_x = self.manual_offset_x
            self.offset_y = self.manual_offset_y
        else:
            print("Gagal membuka kamera eksternal.")
            cap = None

        return cap

    def update_frame(self):
        """
        Mengambil frame dari kamera dan memperbarui canvas.
        Jika gagal mengambil frame, menampilkan pesan error.
        """
        if not self.is_frozen and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                print("Gagal mengambil frame dari kamera.")
            else:
                # Terapkan zoom dan pan
                frame = self.apply_zoom_and_pan(frame)

                # Terapkan mode hitam-putih jika diperlukan
                if self.mode == 'gray':
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)

                # Perbarui frame pada canvas
                self.canvas.itemconfig(self.image_on_canvas, image=imgtk)
                self.canvas.image = imgtk

        # Panggil fungsi ini kembali setelah 10 ms
        self.after(10, self.update_frame)

    def apply_zoom_and_pan(self, frame):
        """
        Terapkan zoom dan geser (pan) ke frame.
        """
        h, w, _ = frame.shape

        # Hitung ukuran zoom
        new_w = int(w * self.zoom_scale)
        new_h = int(h * self.zoom_scale)

        # Resize frame untuk zoom
        frame = cv2.resize(frame, (new_w, new_h))

        # Terapkan geser (pan) dengan offset X dan Y
        start_x = int(max(0, self.offset_x))
        start_y = int(max(0, self.offset_y))

        # Update logic to ensure the frame can be moved fully to the right or bottom
        end_x = int(min(new_w, start_x + 1280))  # batas kanan gambar (untuk 1280px lebar)
        end_y = int(min(new_h, start_y + 720))  # batas bawah gambar (untuk 720px tinggi)

        # Adjust offsets to prevent frame from going out of bounds
        self.offset_x = min(new_w - 1280, self.offset_x)  # ensure we don't exceed right boundary
        self.offset_y = min(new_h - 720, self.offset_y)   # ensure we don't exceed bottom boundary

        # Crop frame sesuai dengan offset dan ukuran yang diinginkan
        cropped_frame = frame[start_y:end_y, start_x:end_x]
        return cropped_frame

    def keypress_event(self, event):
        """
        Tangani input keyboard untuk zoom, geser (pan), dan mode warna/hitam-putih.
        """
        if event.keysym == 'i':  # Zoom in
            self.zoom_scale = min(4.5, self.zoom_scale + 0.1)  # Zoom hingga 4.5x
        elif event.keysym == 'o':  # Zoom out
            self.zoom_scale = max(1.0, self.zoom_scale - 0.1)
        elif event.keysym == 'w':  # Pan up
            self.offset_y = max(0, self.offset_y - 20)
        elif event.keysym == 's':  # Pan down
            self.offset_y = min(int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.zoom_scale - 720), self.offset_y + 20)
        elif event.keysym == 'a':  # Pan left
            self.offset_x = max(0, self.offset_x - 20)
        elif event.keysym == 'd':  # Pan right
            self.offset_x = min(int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.zoom_scale - 1280), self.offset_x + 20)
        elif event.keysym == 'c':  # Switch to color mode
            self.mode = 'color'
            print("Mode diubah ke warna")
        elif event.keysym == 'g':  # Switch to grayscale mode
            self.mode = 'gray'
            print("Mode diubah ke hitam-putih")

    def freeze_frame(self):
        if not self.is_frozen:
            self.is_frozen = True
            print("Live feed frozen")

    def reset_live_feed(self):
        if self.is_frozen:
            self.is_frozen = False
            print("Live feed restarted")

    def finish_capture(self, force_freeze=False):
        """
        Capture image and save it in the folder created by DataPasienScreen.
        """
        if force_freeze:  # Jika force_freeze True, kita set is_frozen menjadi True
            self.is_frozen = True
            print("Frame frozen manually (through 'Selesai' button)")

        if self.is_frozen and self.cap:
            ret, frame = self.cap.read()
            if ret:
                folder_name = self.controller.PATIENTS_DATA_FOLDER  # Folder dari DataPasienScreen

                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                # Simpan gambar asli
                original_image_path = os.path.join(folder_name, "captured-image-original.png")
                cv2.imwrite(original_image_path, frame)
                print(f"Original image saved to {original_image_path}")

                # Crop gambar
                cropped_frame = self.apply_crop(frame)

                # Simpan gambar cropped RGB
                cropped_image_path = os.path.join(folder_name, "captured-image-cropped.png")
                cv2.imwrite(cropped_image_path, cropped_frame)
                print(f"Cropped RGB image saved to {cropped_image_path}")

                # Simpan versi hitam-putih dari gambar cropped
                cropped_bw_path = self.save_black_and_white_image(cropped_frame, folder_name)

                # Melakukan segmentasi YOLOv8 pada gambar cropped BW
                if cropped_bw_path:
                    self.controller.frames["LoadingScreen"].start_segmentation(cropped_bw_path, folder_name)
                    self.controller.show_frame("LoadingScreen")

            self.cap.release()

    def apply_crop(self, frame):
        """
        Crop the frame based on predefined coordinates.
        """
        start_x = self.crop_x
        start_y = self.crop_y
        end_x = start_x + self.crop_width
        end_y = start_y + self.crop_height

        # Validasi agar cropping tidak melebihi dimensi frame
        frame_height, frame_width = frame.shape[:2]
        if end_x > frame_width or end_y > frame_height:
            print("Error: Area cropping melebihi dimensi frame.")
            return frame  # Atau tangani sesuai kebutuhan

        cropped_frame = frame[start_y:end_y, start_x:end_x]
        return cropped_frame

    def save_black_and_white_image(self, image, folder_name):
        """
        Simpan versi hitam-putih dari gambar dan kembalikan path gambar.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # Ubah ke BGR untuk konsistensi
        file_path_gray = os.path.join(folder_name, "captured-image-bw-cropped.png")
        cv2.imwrite(file_path_gray, gray_image_bgr)
        print(f"Black and white cropped image saved to {file_path_gray}")
        return file_path_gray

    def monitor_push_button(self):
        """
        Monitor status push button dan freeze frame ketika push button ditekan.
        """
        while True:
            status = self.push_button_reader.read_push_button_status()
            if status == 'short_press' and not self.is_frozen:
                self.freeze_frame()
            threading.Event().wait(0.1)  # Tunggu 100 ms sebelum mengecek lagi

    # Fungsi untuk mendapatkan nomor pasien terakhir berdasarkan jumlah folder di Data_Pasien
    @staticmethod
    def get_last_patient_number():
        # Menghitung jumlah folder yang ada di dalam folder Data_Pasien
        patient_folders = [f for f in PATIENTS_DATA_FOLDER.iterdir() if f.is_dir()]
        return len(patient_folders)