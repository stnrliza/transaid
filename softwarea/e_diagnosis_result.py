import ctypes
import sqlite3
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
from pathlib import Path

# Path untuk aset
ASSETS_PATH = Path(__file__).resolve().parent / "assets" / "frame-e"

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Mengaktifkan DPI Awareness
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception as e:
    print(f"Error setting DPI awareness: {e}")

class DiagnosisResultScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.conn = sqlite3.connect('pasien.db')
        self.c = self.conn.cursor()

        self.configure(bg="#FFFFFF")

        # Membuat container untuk header
        header_frame = tk.Frame(self, bg="#A8DFE6", height=120)
        header_frame.pack(fill="x", side="top")

        # Header Canvas untuk Judul
        canvas = Canvas(header_frame, bg="#A8DFE6", height=104, width=1920, bd=0, highlightthickness=0, relief="ridge")
        canvas.pack(fill="x")
        canvas.create_text(85, 20, anchor="nw", text="Hasil Pemeriksaan", fill="#15218E", font=("Poppins Bold", 43 * -1))

        # Placeholder untuk hasil gambar segmentasi
        segmented_image_frame = tk.Frame(self, bg="#FFFFFF")
        segmented_image_frame.pack(pady=(50, 0))

        self.segmented_image_label = tk.Label(segmented_image_frame, bg="#FFFFFF")
        self.segmented_image_label.pack()

        # Tombol untuk memuat hasil segmentasi
        button_frame = tk.Frame(self, bg="#FFFFFF")
        button_frame.pack(pady=20)

        button_load = tk.Button(button_frame, text="Muat Hasil Segmentasi", font=("Poppins", 16),
                                command=self.load_segmented_image, bg="#A8DFE6", fg="#15218E", relief="raised")
        button_load.pack()

    def load_segmented_image(self, image_path=None):
        """
        Memuat dan menampilkan gambar hasil segmentasi dari path yang diberikan.
        Jika tidak ada path yang diberikan, gambar terbaru akan diambil dari database.
        """
        if image_path is None:
            # Mengambil path gambar dari database
            self.c.execute("SELECT path_segmentasi FROM pasien ORDER BY id DESC LIMIT 1")
            row = self.c.fetchone()
            image_path = row[0] if row else None

        if image_path and Path(image_path).exists():
            img = Image.open(image_path)
            imgtk = ImageTk.PhotoImage(img)
            self.segmented_image_label.configure(image=imgtk)
            self.segmented_image_label.image = imgtk
            print(f"Gambar hasil segmentasi dimuat dari {image_path}")
        else:
            print("Tidak ada gambar valid yang ditemukan.")
