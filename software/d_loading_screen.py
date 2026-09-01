import os
import sqlite3
import threading
import tkinter as tk
from tkinter import Canvas, messagebox
import customtkinter as ctk
from pathlib import Path
from PIL import Image, ImageTk
from yolov8_segment import run_yolov8_segmentation

ASSETS_PATH = Path(__file__).resolve().parent / "assets" / "frame-d"
DATABASE_PATH = Path(__file__).resolve().parent / "pasien.db"

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class LoadingScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#FFFFFF")

        self.image_path = None
        self.patient_folder = None
        self.progress_value = 0.0
        self.segmentation_done = False
        self.segmentation_error = None
        self.output_path = None

        # Container
        self.container = tk.Frame(self, bg="#FFFFFF")
        self.container.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Header Image / Logo
        self.image_label = tk.Label(self.container, bg="#FFFFFF")
        self.image_label.place(relx=0.5, rely=0.25, anchor="center")

        image_path = relative_to_assets("d-image.png")
        if image_path.exists():
            img = Image.open(image_path)
            # Resize image reasonably
            img_resized = img.resize((320, 200), Image.LANCZOS)
            self.header_photo = ImageTk.PhotoImage(img_resized)
            self.image_label.configure(image=self.header_photo)

        # Title Label
        self.message_label = ctk.CTkLabel(
            self.container,
            text="Sedang Memproses Segmentasi...",
            font=("Poppins Bold", 32),
            text_color="#15218E"
        )
        self.message_label.place(relx=0.5, rely=0.48, anchor="center")

        # Subtitle / Status text
        self.status_label = ctk.CTkLabel(
            self.container,
            text="Menganalisis karies gigi dengan model AI YOLOv8...",
            font=("Poppins", 15),
            text_color="#555555"
        )
        self.status_label.place(relx=0.5, rely=0.54, anchor="center")

        # CustomTkinter Progress Bar
        self.progress_bar = ctk.CTkProgressBar(
            self.container,
            width=500,
            height=20,
            corner_radius=10,
            progress_color="#15218E",
            fg_color="#E0E6ED"
        )
        self.progress_bar.place(relx=0.5, rely=0.62, anchor="center")
        self.progress_bar.set(0.0)

        # Percentage label
        self.pct_label = ctk.CTkLabel(
            self.container,
            text="0%",
            font=("Poppins Bold", 16),
            text_color="#15218E"
        )
        self.pct_label.place(relx=0.5, rely=0.67, anchor="center")

    def on_show(self):
        """Reset progress state saat layar ditampilkan."""
        self.progress_value = 0.0
        self.segmentation_done = False
        self.segmentation_error = None
        self.progress_bar.set(0.0)
        self.pct_label.configure(text="0%")
        self.message_label.configure(text="Sedang Memproses Segmentasi...")
        self.status_label.configure(text="Menganalisis karies gigi dengan model AI YOLOv8...")

    def start_segmentation(self, image_path, patient_folder):
        """
        Memulai proses segmentasi di background thread dan animasi progress bar di main thread.
        """
        self.image_path = image_path
        self.patient_folder = patient_folder
        self.output_path = str(Path(patient_folder) / "output-segmented.jpg")
        self.progress_value = 0.0
        self.segmentation_done = False
        self.segmentation_error = None

        # Jalankan background worker untuk inferensi AI
        worker_thread = threading.Thread(target=self._run_segmentation_worker, daemon=True)
        worker_thread.start()

        # Jalankan loop animasi progress bar pada main thread
        self._animate_progress()

    def _run_segmentation_worker(self):
        """Worker thread untuk menjalankan YOLOv8 tanpa memblokir GUI."""
        try:
            print(f"[LoadingScreen] Menjalankan YOLOv8 pada: {self.image_path}")
            run_yolov8_segmentation(self.image_path, self.output_path)
            self.segmentation_done = True
        except Exception as e:
            print(f"[LoadingScreen] Error during YOLOv8 segmentation: {e}")
            self.segmentation_error = str(e)
            self.segmentation_done = True

    def _animate_progress(self):
        """Animasi progress bar halus yang terkoordinasi dengan status thread."""
        if not self.segmentation_done:
            # Tingkatkan progress bar perlahan hingga 85% sambil menunggu model
            if self.progress_value < 0.85:
                self.progress_value += 0.03
            self.progress_bar.set(self.progress_value)
            self.pct_label.configure(text=f"{int(self.progress_value * 100)}%")
            self.after(50, self._animate_progress)
        else:
            # Model selesai, isi hingga 100% lalu beralih
            if self.progress_value < 1.0:
                self.progress_value = min(1.0, self.progress_value + 0.1)
                self.progress_bar.set(self.progress_value)
                self.pct_label.configure(text=f"{int(self.progress_value * 100)}%")
                self.after(30, self._animate_progress)
            else:
                # Selesai 100% -> proses hasil pada main thread
                self._handle_completion()

    def _handle_completion(self):
        """Dipanggil di main thread setelah animasi 100% selesai."""
        if self.segmentation_error:
            # Jika ada error pada model AI
            messagebox.showwarning(
                "Peringatan Model AI",
                f"Proses segmentasi mengalami kendala:\n{self.segmentation_error}\n\nAplikasi akan menampilkan gambar hasil tangkapan kamera."
            )
            # Fallback output ke captured image
            fallback_path = self.image_path
            self.output_path = fallback_path

        # Simpan path_segmentasi ke database
        patient_id = getattr(self.controller, 'current_patient_id', None)
        if patient_id and DATABASE_PATH.exists():
            try:
                conn = sqlite3.connect(DATABASE_PATH)
                c = conn.cursor()
                c.execute("UPDATE pasien SET path_segmentasi = ? WHERE id = ?", (str(self.output_path), patient_id))
                conn.commit()
                conn.close()
                print(f"[LoadingScreen] Database diperbarui: path_segmentasi = {self.output_path}")
            except Exception as e:
                print(f"[LoadingScreen] Error updating database: {e}")

        # Simpan ke controller untuk ditampilkan di DiagnosisResultScreen
        self.controller.segmentation_output_path = self.output_path

        # Pindah ke layar hasil diagnosis
        self.controller.show_frame("DiagnosisResultScreen")
