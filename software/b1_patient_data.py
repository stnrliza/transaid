import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import sqlite3
from pathlib import Path
from PIL import Image, ImageTk
from datetime import datetime

ASSETS_PATH = Path(__file__).resolve().parent / "assets" / "frame-b1"
PATIENTS_DATA_FOLDER = Path(__file__).resolve().parent / "Data_Pasien"
DATABASE = Path(__file__).resolve().parent / "pasien.db"

PATIENTS_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / path

class PatientDataScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#FFFFFF")

        # Inisialisasi database tabel pasien dengan skema lengkap
        self.init_database()

        # Frame container
        self.container = tk.Frame(self, bg="#FFFFFF")
        self.container.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Logo TransAID
        logo_path = relative_to_assets("b1-image.png")
        if logo_path.exists():
            logo_image = Image.open(logo_path)
            logo_w = max(100, int(screen_width / 9.6))
            logo_h = max(60, int(screen_height / 5.4))
            responsive_logo = logo_image.resize((logo_w, logo_h), Image.LANCZOS)
            self.logo_img = ImageTk.PhotoImage(responsive_logo)
            self.logo_label = tk.Label(self.container, image=self.logo_img, bg="#FFFFFF")
            self.logo_label.place(relx=0.5, rely=0.1, anchor="n")
        else:
            self.logo_label = ctk.CTkLabel(
                self.container,
                text="TransAID",
                font=("Poppins Bold", 36),
                text_color="#16228E"
            )
            self.logo_label.place(relx=0.5, rely=0.1, anchor="n")

        self.font_size = max(14, int(screen_height / 48))

        # Label & Entry: Nama Pasien
        self.name_entry_label = ctk.CTkLabel(
            self.container,
            text="Nama Pasien",
            font=("Poppins Bold", self.font_size),
            text_color="#000000",
            fg_color="transparent"
        )
        self.name_entry_label.place(relx=0.1, rely=0.38, anchor="nw")

        self.name_entry = ctk.CTkEntry(
            self.container,
            placeholder_text="Masukkan nama lengkap pasien...",
            fg_color="#F0F4F8",
            text_color="#000000",
            font=("Poppins Medium", self.font_size),
            corner_radius=15,
            height=45
        )
        self.name_entry.place(relx=0.1, rely=0.44, anchor="nw", relwidth=0.8)

        # Label & Entry: Tanggal Pemeriksaan
        self.date_entry_label = ctk.CTkLabel(
            self.container,
            text="Tanggal Pemeriksaan (YYYY-MM-DD)",
            font=("Poppins Bold", self.font_size),
            fg_color="transparent",
            text_color="#000000"
        )
        self.date_entry_label.place(relx=0.1, rely=0.56, anchor="nw")

        self.date_entry = ctk.CTkEntry(
            self.container,
            fg_color="#F0F4F8",
            text_color="#000000",
            font=("Poppins Medium", self.font_size),
            corner_radius=15,
            height=45
        )
        self.date_entry.place(relx=0.1, rely=0.62, anchor="nw", relwidth=0.8)
        self.date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))

        # Tombol Kembali
        back_button = ctk.CTkButton(
            self.container,
            text="Kembali",
            font=("Poppins Medium", self.font_size),
            fg_color="#A8DEE6",
            text_color="#16228E",
            hover_color="#8ecbd4",
            corner_radius=15,
            width=150,
            height=45,
            command=lambda: controller.show_frame("TransAIDScreen")
        )
        back_button.place(relx=0.25, rely=0.8, anchor="center")

        # Tombol Selanjutnya
        save_button = ctk.CTkButton(
            self.container,
            text="Selanjutnya",
            font=("Poppins Medium", self.font_size),
            fg_color="#16228E",
            text_color="#FFFFFF",
            hover_color="#0e1761",
            corner_radius=15,
            width=150,
            height=45,
            command=self.save_and_navigate
        )
        save_button.place(relx=0.75, rely=0.8, anchor="center")

    def init_database(self):
        """Membuat tabel pasien dengan skema lengkap bila belum ada."""
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS pasien (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama TEXT NOT NULL,
            tanggal_pemeriksaan TEXT NOT NULL,
            path_gambar TEXT,
            path_segmentasi TEXT
        )''')
        conn.commit()
        conn.close()

    def on_show(self):
        """Dipanggil saat frame ditampilkan untuk refresh form."""
        self.name_entry.delete(0, tk.END)
        self.date_entry.delete(0, tk.END)
        self.date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))
        self.name_entry.focus_set()

    def save_and_navigate(self):
        """
        Menyimpan data pasien ke database dan membuat folder pasien,
        lalu berpindah ke LiveCameraScreen.
        """
        nama = self.name_entry.get().strip()
        tanggal_pemeriksaan = self.date_entry.get().strip()

        if not nama:
            messagebox.showwarning("Data Tidak Lengkap", "Nama pasien tidak boleh kosong!")
            return

        # Validasi format tanggal YYYY-MM-DD
        try:
            tanggal_pemeriksaan = datetime.strptime(tanggal_pemeriksaan, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Format Tanggal Salah", "Format tanggal tidak valid!\nSilakan gunakan format YYYY-MM-DD (contoh: 2026-09-01).")
            return

        # Menyimpan data pasien ke database
        patient_id = self.insert_pasien(nama, tanggal_pemeriksaan)

        # Membuat folder unik untuk pasien ini
        folder_name = PATIENTS_DATA_FOLDER / f"{tanggal_pemeriksaan}_Pasien-{patient_id}"
        folder_name.mkdir(parents=True, exist_ok=True)
        print(f"Folder pasien dibuat: {folder_name}")

        # Simpan state ke controller agar dapat diakses oleh LiveCameraScreen & DiagnosisResultScreen
        self.controller.current_patient_id = patient_id
        self.controller.current_patient_nama = nama
        self.controller.current_patient_tanggal = tanggal_pemeriksaan
        self.controller.current_patient_folder = folder_name

        # Pindah ke LiveCameraScreen
        self.controller.show_frame("LiveCameraScreen")

    def insert_pasien(self, nama, tanggal_pemeriksaan):
        """Menyimpan data pasien baru ke database dan mengembalikan ID-nya."""
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("INSERT INTO pasien (nama, tanggal_pemeriksaan) VALUES (?, ?)", (nama, tanggal_pemeriksaan))
        conn.commit()
        patient_id = c.lastrowid
        conn.close()
        return patient_id
