import sqlite3
import tkinter as tk
from tkinter import Canvas, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
from pathlib import Path

# Path konstanta
ASSETS_PATH = Path(__file__).resolve().parent / "assets" / "frame-e"
DATABASE_PATH = Path(__file__).resolve().parent / "pasien.db"

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class DiagnosisResultScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#FFFFFF")

        self.current_image_path = None
        self.photo_ref = None

        # Container utama
        self.container = tk.Frame(self, bg="#FFFFFF")
        self.container.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Header Bar
        self.header_frame = tk.Frame(self.container, bg="#A8DFE6", height=90)
        self.header_frame.pack(fill="x", side="top")

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Hasil Diagnosis Karies Gigi",
            font=("Poppins Bold", 26),
            text_color="#15218E"
        )
        self.title_label.place(relx=0.04, rely=0.5, anchor="w")

        # Patient Info Header Badges
        self.patient_info_label = ctk.CTkLabel(
            self.header_frame,
            text="Pasien: - | Tanggal: -",
            font=("Poppins Medium", 14),
            text_color="#16228E"
        )
        self.patient_info_label.place(relx=0.55, rely=0.5, anchor="w")

        # Tombol Beranda di Header
        self.home_button = ctk.CTkButton(
            self.header_frame,
            text="Beranda",
            font=("Poppins Medium", 14),
            fg_color="#16228E",
            text_color="#FFFFFF",
            hover_color="#0e1761",
            corner_radius=10,
            width=100,
            height=36,
            command=lambda: controller.show_frame("TransAIDScreen")
        )
        self.home_button.place(relx=0.94, rely=0.5, anchor="e")

        # Image Display Canvas
        self.image_frame = tk.Frame(self.container, bg="#F5F7FA")
        self.image_frame.pack(fill="both", expand=True, padx=30, pady=(15, 10))

        self.canvas = Canvas(self.image_frame, bg="#F5F7FA", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="center")

        # Placeholder text on canvas
        self.placeholder_text_id = self.canvas.create_text(
            0, 0, anchor="center",
            text="Memuat Gambar Hasil Segmentasi...",
            fill="#7F8C8D",
            font=("Poppins", 16)
        )

        # Bottom Action Bar
        self.bottom_bar = tk.Frame(self.container, bg="#FFFFFF", height=70)
        self.bottom_bar.pack(fill="x", side="bottom", padx=30, pady=(0, 15))

        # Tombol Riwayat
        self.history_button = ctk.CTkButton(
            self.bottom_bar,
            text="📋 Riwayat Pasien",
            font=("Poppins Medium", 14),
            fg_color="#A8DFE6",
            text_color="#16228E",
            hover_color="#8ecbd4",
            corner_radius=12,
            width=170,
            height=44,
            command=lambda: controller.show_frame("DiagnosisHistoryScreen")
        )
        self.history_button.pack(side="left", padx=(0, 10))

        # Tombol Muat Ulang
        self.reload_button = ctk.CTkButton(
            self.bottom_bar,
            text="🔄 Muat Ulang",
            font=("Poppins Medium", 14),
            fg_color="#E0E6ED",
            text_color="#333333",
            hover_color="#d0d8e2",
            corner_radius=12,
            width=140,
            height=44,
            command=self.load_segmented_image
        )
        self.reload_button.pack(side="left", padx=5)

        # Tombol Pemeriksaan Baru
        self.new_exam_button = ctk.CTkButton(
            self.bottom_bar,
            text="+ Pemeriksaan Baru",
            font=("Poppins Bold", 15),
            fg_color="#16228E",
            text_color="#FFFFFF",
            hover_color="#0e1761",
            corner_radius=12,
            width=200,
            height=44,
            command=lambda: controller.show_frame("PatientDataScreen")
        )
        self.new_exam_button.pack(side="right")

    def on_show(self):
        """Dipanggil otomatis saat layar hasil ditampilkan."""
        # Update badge nama & tanggal pasien
        nama = getattr(self.controller, 'current_patient_nama', None)
        tanggal = getattr(self.controller, 'current_patient_tanggal', None)

        if not nama:
            # Coba ambil pasien terakhir dari database
            last_record = self.get_latest_patient_record()
            if last_record:
                nama = last_record.get("nama", "-")
                tanggal = last_record.get("tanggal", "-")

        info_str = f"Pasien: {nama or '-'} | Tanggal: {tanggal or '-'}"
        self.patient_info_label.configure(text=info_str)

        # Muat gambar segmentasi
        self.after(50, self.load_segmented_image)

    def get_latest_patient_record(self):
        """Mengambil data pasien terbaru dari database."""
        if not DATABASE_PATH.exists():
            return None
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            c = conn.cursor()
            c.execute("SELECT nama, tanggal_pemeriksaan, path_segmentasi, path_gambar FROM pasien ORDER BY id DESC LIMIT 1")
            row = c.fetchone()
            conn.close()
            if row:
                return {
                    "nama": row[0],
                    "tanggal": row[1],
                    "path_segmentasi": row[2],
                    "path_gambar": row[3]
                }
        except Exception as e:
            print(f"Error fetching latest patient record: {e}")
        return None

    def load_segmented_image(self, image_path=None):
        """
        Memuat dan menampilkan gambar hasil segmentasi.
        Jika image_path tidak dispesifikasikan, menggunakan path dari controller atau database.
        """
        if image_path is None:
            image_path = getattr(self.controller, 'segmentation_output_path', None)

        if image_path is None:
            latest = self.get_latest_patient_record()
            if latest:
                image_path = latest.get("path_segmentasi") or latest.get("path_gambar")

        self.current_image_path = image_path

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 10:
            cw = 1200
        if ch <= 10:
            ch = 600

        if image_path and Path(image_path).exists():
            try:
                img = Image.open(image_path)
                iw, ih = img.size
                scale = min(cw / iw, ch / ih) * 0.95
                tw = max(1, int(iw * scale))
                th = max(1, int(ih * scale))

                img_resized = img.resize((tw, th), Image.LANCZOS)
                self.photo_ref = ImageTk.PhotoImage(img_resized)

                self.canvas.itemconfig(self.image_on_canvas, image=self.photo_ref)
                self.canvas.coords(self.image_on_canvas, cw // 2, ch // 2)
                self.canvas.itemconfig(self.placeholder_text_id, text="")
                print(f"[DiagnosisResult] Gambar berhasil dimuat dari: {image_path}")
            except Exception as e:
                print(f"[DiagnosisResult] Error rendering image: {e}")
                self.canvas.itemconfig(self.placeholder_text_id, text=f"Gagal memuat gambar:\n{e}")
        else:
            self.canvas.itemconfig(self.placeholder_text_id, text="Belum ada gambar hasil segmentasi yang tersimpan.")
            self.canvas.coords(self.placeholder_text_id, cw // 2, ch // 2)
