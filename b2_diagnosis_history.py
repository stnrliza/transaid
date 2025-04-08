import ctypes
import customtkinter as ctk
import sqlite3
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

PATIENT_DATA_PATH = Path(__file__).parent / "Data_Pasien"

def relative_to_patient_data(path: str) -> Path:
    return PATIENT_DATA_PATH / path

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception as e:
    print(f"Error setting DPI awareness: {e}")

class DiagnosisHistoryScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#FFFFFF")

        # Container utama
        self.container = tk.Frame(self, bg="#A8DFE6")
        self.container.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Header
        self.header_frame = tk.Frame(self.container, bg="#A8DFE6")
        self.header_frame.place(relx=0, rely=0, relwidth=1, relheight=0.15)

        self.subtitle_label = ctk.CTkLabel(
            self.header_frame, text="Daftar Riwayat Pemeriksaan",
            font=("Poppins Bold", 24), text_color="#16228E"
        )
        self.subtitle_label.place(relx=0.05, rely=0.3, anchor="w")

        self.search_var = tk.StringVar()
        self.search_entry = ctk.CTkEntry(
            self.header_frame, textvariable=self.search_var,
            font=("Poppins", 16), width=250, placeholder_text="Cari Nama Pasien...",
            text_color="#000000",
            fg_color="#DDDDDD"
        )
        self.search_entry.place(relx=0.65, rely=0.3, anchor="w")
        self.search_entry.bind("<KeyRelease>", self.search_patient)

        self.back_button = ctk.CTkButton(
            self.header_frame, text="Kembali",
            font=("Poppins Medium", 16), fg_color="#16228E", text_color="white",
            corner_radius=15, command=lambda: controller.show_frame("TransAIDScreen")
        )
        self.back_button.place(relx=0.85, rely=0.3, anchor="w")

        # Container untuk tabel
        self.table_container = tk.Frame(self.container, bg="#FFFFFF")
        self.table_container.place(relx=0.05, rely=0.2, relwidth=0.9, relheight=0.7)

        # Tabel Treeview
        self.tree = ttk.Treeview(
            self.table_container, columns=("No.", "Nama Pasien", "Tanggal Pemeriksaan", "Hasil Pemeriksaan"),
            show='headings', selectmode='browse'
        )
        self.tree.place(relx=0, rely=0, relwidth=0.95, relheight=0.9)

        # Mengatur gaya Treeview agar font lebih besar
        style = ttk.Style()
        style.configure("Treeview", font=("Poppins", 24))  # Ubah ukuran font di sini
        style.configure("Treeview.Heading", font=("Poppins Bold", 20))  # Untuk heading
        style.configure("Treeview", rowheight=50)  # Atur tinggi baris

        # Heading
        self.tree.heading("No.", text="No.")
        self.tree.heading("Nama Pasien", text="Nama Pasien")
        self.tree.heading("Tanggal Pemeriksaan", text="Tanggal Pemeriksaan")
        self.tree.heading("Hasil Pemeriksaan", text="Hasil Pemeriksaan")

        self.tree.column("No.", width=50, anchor='center')
        self.tree.column("Nama Pasien", width=400, anchor='w')
        self.tree.column("Tanggal Pemeriksaan", width=200, anchor='center')
        self.tree.column("Hasil Pemeriksaan", width=200, anchor='center')

        # Scrollbar
        self.scrollbar = ttk.Scrollbar(self.table_container, orient="vertical", command=self.tree.yview)
        self.scrollbar.place(relx=0.95, rely=0, relwidth=0.05, relheight=0.9)
        self.tree.configure(yscrollcommand=self.scrollbar.set)

        # Pagination
        self.pagination_frame = tk.Frame(self.table_container, bg="#FFFFFF")
        self.pagination_frame.place(relx=0, rely=0.92, relwidth=1, relheight=0.08)

        self.prev_button = ctk.CTkButton(
            self.pagination_frame, text="<< Prev",
            font=("Poppins Medium", 14), fg_color="#16228E", text_color="white",
            corner_radius=10, command=self.prev_page
        )
        self.prev_button.place(relx=0.3, rely=0.5, anchor="center")

        self.page_label = ctk.CTkLabel(
            self.pagination_frame, text="Page 1",
            font=("Poppins", 14), text_color="#16228E"
        )
        self.page_label.place(relx=0.5, rely=0.5, anchor="center")

        self.next_button = ctk.CTkButton(
            self.pagination_frame, text="Next >>",
            font=("Poppins Medium", 14), fg_color="#16228E", text_color="white",
            corner_radius=10, command=self.next_page
        )
        self.next_button.place(relx=0.7, rely=0.5, anchor="center")

        # Event double-click
        self.tree.bind("<Double-1>", self.open_diagnosis_result)

        # Data pasien
        self.page = 1
        self.items_per_page = 5
        self.total_pages = 1
        self.data_list = []
        self.filtered_data = []

        self.display_folders()

    def open_diagnosis_result(self, event):
        """ Event saat pasien diklik dua kali untuk melihat hasil diagnosis """
        selected_item = self.tree.selection()
        if not selected_item:
            return

        item = self.tree.item(selected_item)
        values = item["values"]

        if not values:
            return

        nama_pasien = values[1]
        tanggal_pemeriksaan = values[2]

        self.controller.show_frame("DiagnosisResultScreen", nama_pasien=nama_pasien, tanggal=tanggal_pemeriksaan)

    def search_patient(self, event=None):
        """ Filter daftar pasien berdasarkan input pencarian """
        search_query = self.search_var.get().strip().lower()

        if not search_query:
            self.filtered_data = self.data_list
        else:
            self.filtered_data = [
                row for row in self.data_list if search_query in row[1].lower()
            ]

        self.total_pages = (len(self.filtered_data) // self.items_per_page) + (1 if len(self.filtered_data) % self.items_per_page != 0 else 0)
        self.page = 1
        self.update_table()

    def display_folders(self):
        self.data_list = []

        if not PATIENT_DATA_PATH.exists():
            print(f"Folder data pasien tidak ditemukan: {PATIENT_DATA_PATH}")
            return

        folder_list = [folder for folder in PATIENT_DATA_PATH.iterdir() if folder.is_dir()]

        def sort_key(folder_name):
            folder_parts = folder_name.name.split('_')
            tanggal = folder_parts[0]
            pasien = folder_parts[1]
            pasien_num = int(pasien.split('-')[1])
            return (datetime.strptime(tanggal, '%Y-%m-%d'), pasien_num)

        folder_list.sort(key=sort_key, reverse=True)

        for idx, folder in enumerate(folder_list, start=1):
            folder_parts = folder.name.split('_')
            tanggal_pemeriksaan = folder_parts[0]
            nama_pasien = folder_parts[1]
            self.data_list.append((idx, nama_pasien, tanggal_pemeriksaan, ""))

        self.filtered_data = self.data_list

        self.total_pages = (len(self.data_list) // self.items_per_page) + (1 if len(self.data_list) % self.items_per_page != 0 else 0)
        self.page = 1
        self.update_table()

    def update_table(self):
        self.tree.delete(*self.tree.get_children())

        start = (self.page - 1) * self.items_per_page
        end = start + self.items_per_page
        for row in self.filtered_data[start:end]:
            self.tree.insert("", "end", values=row)

        self.page_label.configure(text=f"Page {self.page} of {self.total_pages}")
        self.prev_button.configure(state="normal" if self.page > 1 else "disabled")
        self.next_button.configure(state="normal" if self.page < self.total_pages else "disabled")

    def prev_page(self):
        """Navigasi ke halaman sebelumnya"""
        if self.page > 1:
            self.page -= 1
            self.update_table()

    def next_page(self):
        """Navigasi ke halaman berikutnya"""
        if self.page < self.total_pages:
            self.page += 1
            self.update_table()