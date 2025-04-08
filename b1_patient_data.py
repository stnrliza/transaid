import ctypes
import customtkinter as ctk
import tkinter as tk
import sqlite3
from pathlib import Path
from PIL import Image, ImageTk
from datetime import datetime  

ASSETS_PATH = Path(__file__).parent / "assets" / "frame-b1"
PATIENTS_DATA_FOLDER = Path(__file__).parent / "Data_Pasien"
DATABASE = Path(__file__).parent / "pasien.db"

PATIENTS_DATA_FOLDER.mkdir(parents=True, exist_ok=True)  # Membuat folder data pasien jika belum ada

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / path

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception as e:
    print(f"Error setting DPI awareness: {e}")

class PatientDataScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#FFFFFF")

        # Koneksi ke SQLite
        self.conn = sqlite3.connect(DATABASE)
        self.c = self.conn.cursor()
        self.c.execute('''CREATE TABLE IF NOT EXISTS pasien
                          (id INTEGER PRIMARY KEY, nama TEXT, tanggal_pemeriksaan TEXT)''')

        # Frame untuk menempatkan elemen
        self.container = tk.Frame(self, bg="#FFFFFF")
        self.container.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Menambahkan logo (lebih kecil dari yang ada di welcome screen)
        logo_path = relative_to_assets("b1-image.png")
        if logo_path.exists():
            logo_image = Image.open(logo_path)
            responsive_logo_size = logo_image.resize((int(screen_width/9.6), int(screen_height/5.4)), Image.LANCZOS)
            logo = ImageTk.PhotoImage(responsive_logo_size)
            self.logo_label = tk.Label(self.container, image=logo, bg="#FFFFFF")
            self.logo_label.image = logo
            self.logo_label.place(relx=0.5, rely=0.1, anchor="n")
        else:
            print(f"Error: File tidak ditemukan - {logo_path}")

        # Label untuk entry nama pasien
        self.font_size = int(screen_height / 48)

        self.name_entry_label = ctk.CTkLabel(
            self.container,
            text="Nama Pasien",
            font=("Poppins Bold", self.font_size),
            text_color="#000000",
            fg_color="transparent")
        self.name_entry_label.place(relx=0.1, rely=0.4, anchor="nw")

        # Entry nama pasien
        self.name_entry = ctk.CTkEntry(
            self.container,
            fg_color="#DDDDDD",
            text_color="#000000",
            font=("Poppins Medium", self.font_size),
            corner_radius=15)
        self.name_entry.place(relx=0.1, rely=0.45, anchor="nw", relwidth=0.8)

        # Label untuk entry tanggal pemeriksaan
        self.date_entry_label = ctk.CTkLabel(
            self.container,
            text="Tanggal Pemeriksaan",
            font=("Poppins Bold", self.font_size),
            fg_color="transparent",
            text_color="#000000")
        self.date_entry_label.place(relx=0.1, rely=0.6, anchor="nw")

        # Entry tanggal pemeriksaan
        self.date_entry = ctk.CTkEntry(
            self.container,
            fg_color="#DDDDDD",
            text_color="#000000",
            font=("Poppins Medium", self.font_size),
            corner_radius=15)
        self.date_entry.place(relx=0.1, rely=0.65, anchor="nw", relwidth=0.8)

        # Tombol simpan dan navigasi ke LiveCameraScreen
        back_button = ctk.CTkButton(
            self.container,
            text="Kembali",
            font=("Poppins Medium", self.font_size),
            fg_color="#A8DEE6",
            text_color="#16228E",
            corner_radius=15,
            command=lambda: controller.show_frame("TransAIDScreen")
        )
        back_button.place(relx=0.4, rely=0.8, anchor="nw")

        # Tombol simpan dan navigasi ke LiveCameraScreen
        save_button = ctk.CTkButton(
            self.container,
            text="Selanjutnya",
            font=("Poppins Medium", self.font_size),
            fg_color="#A8DEE6",
            text_color="#16228E",
            corner_radius=15,
            command=self.save_and_navigate  # Memanggil fungsi save_and_navigate sebelum pindah halaman
        )
        save_button.place(relx=0.8, rely=0.8, anchor="nw")

    def save_and_navigate(self):
        """
        Menyimpan data pasien dan membuat folder pasien, lalu berpindah ke LiveCameraScreen.
        """
        nama, tanggal_pemeriksaan = self.get_data_pasien()

        if not nama.strip():  # Memeriksa apakah nama pasien kosong
            print("Nama pasien tidak boleh kosong!")
            return

        # Format tanggal pemeriksaan ke YYYY-MM-DD
        try:
            tanggal_pemeriksaan = datetime.strptime(tanggal_pemeriksaan, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            print("Format tanggal tidak valid! Harus dalam format YYYY-MM-DD.")
            return

        # Menyimpan data ke database
        self.save_pasien(nama, tanggal_pemeriksaan)

        # Ambil nomor pasien terakhir
        patient_number = self.get_last_patient_number() + 1

        # Membuat folder untuk setiap pasien
        folder_name = PATIENTS_DATA_FOLDER / f"{tanggal_pemeriksaan}_Pasien-{patient_number}"
        folder_name.mkdir(parents=True, exist_ok=True)
        print(f"Folder created: {folder_name}")

        # Pindah ke LiveCameraScreen setelah menyimpan data
        self.controller.show_frame("LiveCameraScreen")

    def get_last_patient_number(self):
        """ Mengambil jumlah pasien dari database. """
        self.c.execute("SELECT COUNT(*) FROM pasien")
        return self.c.fetchone()[0]

    def save_pasien(self, nama, tanggal_pemeriksaan):
        """ Menyimpan data pasien ke database. """
        self.c.execute("INSERT INTO pasien (nama, tanggal_pemeriksaan) VALUES (?, ?)", (nama, tanggal_pemeriksaan))
        self.conn.commit()

    def get_data_pasien(self):
        """ Mengambil data pasien dari entry. """
        return self.name_entry.get(), self.date_entry.get()

    def __del__(self):
        """ Menutup koneksi ke database ketika objek dihapus. """
        self.conn.close()