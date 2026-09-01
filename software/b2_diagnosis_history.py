import customtkinter as ctk
import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from pathlib import Path

PATIENT_DATA_PATH = Path(__file__).resolve().parent / "Data_Pasien"
DATABASE_PATH = Path(__file__).resolve().parent / "pasien.db"

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
            font=("Poppins Bold", 26), text_color="#16228E"
        )
        self.subtitle_label.place(relx=0.05, rely=0.5, anchor="w")

        self.search_var = tk.StringVar()
        self.search_entry = ctk.CTkEntry(
            self.header_frame, textvariable=self.search_var,
            font=("Poppins", 15), width=260, height=40,
            placeholder_text="Cari Nama Pasien...",
            text_color="#000000",
            fg_color="#FFFFFF",
            corner_radius=12
        )
        self.search_entry.place(relx=0.62, rely=0.5, anchor="w")
        self.search_entry.bind("<KeyRelease>", self.search_patient)

        self.back_button = ctk.CTkButton(
            self.header_frame, text="Kembali",
            font=("Poppins Medium", 15), fg_color="#16228E", text_color="white",
            hover_color="#0e1761",
            corner_radius=12, width=120, height=40,
            command=lambda: controller.show_frame("TransAIDScreen")
        )
        self.back_button.place(relx=0.88, rely=0.5, anchor="w")

        # Container untuk tabel
        self.table_container = tk.Frame(self.container, bg="#FFFFFF")
        self.table_container.place(relx=0.05, rely=0.17, relwidth=0.9, relheight=0.72)

        # Style Treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Custom.Treeview",
            background="#FFFFFF",
            foreground="#222222",
            fieldbackground="#FFFFFF",
            font=("Poppins", 13),
            rowheight=38
        )
        style.configure(
            "Custom.Treeview.Heading",
            background="#16228E",
            foreground="#FFFFFF",
            font=("Poppins Bold", 13),
            relief="flat"
        )
        style.map("Custom.Treeview", background=[("selected", "#A8DEE6")], foreground=[("selected", "#16228E")])

        # Tabel Treeview
        columns = ("No.", "ID", "Nama Pasien", "Tanggal Pemeriksaan", "Status Diagnosis")
        self.tree = ttk.Treeview(
            self.table_container,
            columns=columns,
            show='headings',
            selectmode='browse',
            style="Custom.Treeview"
        )
        self.tree.place(relx=0.02, rely=0.03, relwidth=0.94, relheight=0.85)

        # Heading & Columns
        self.tree.heading("No.", text="No.")
        self.tree.heading("ID", text="ID")
        self.tree.heading("Nama Pasien", text="Nama Pasien")
        self.tree.heading("Tanggal Pemeriksaan", text="Tanggal Pemeriksaan")
        self.tree.heading("Status Diagnosis", text="Status Hasil")

        self.tree.column("No.", width=50, anchor='center')
        self.tree.column("ID", width=60, anchor='center')
        self.tree.column("Nama Pasien", width=350, anchor='w')
        self.tree.column("Tanggal Pemeriksaan", width=180, anchor='center')
        self.tree.column("Status Diagnosis", width=160, anchor='center')

        # Scrollbar
        self.scrollbar = ttk.Scrollbar(self.table_container, orient="vertical", command=self.tree.yview)
        self.scrollbar.place(relx=0.965, rely=0.03, relwidth=0.02, relheight=0.85)
        self.tree.configure(yscrollcommand=self.scrollbar.set)

        # Bottom Bar: Pagination & Open Result Action
        self.bottom_frame = tk.Frame(self.table_container, bg="#FFFFFF")
        self.bottom_frame.place(relx=0.02, rely=0.90, relwidth=0.96, relheight=0.08)

        self.prev_button = ctk.CTkButton(
            self.bottom_frame, text="<< Prev",
            font=("Poppins Medium", 13), fg_color="#16228E", text_color="white",
            corner_radius=8, width=90, height=32, command=self.prev_page
        )
        self.prev_button.place(relx=0.1, rely=0.5, anchor="center")

        self.page_label = ctk.CTkLabel(
            self.bottom_frame, text="Page 1 of 1",
            font=("Poppins", 13), text_color="#16228E"
        )
        self.page_label.place(relx=0.25, rely=0.5, anchor="center")

        self.next_button = ctk.CTkButton(
            self.bottom_frame, text="Next >>",
            font=("Poppins Medium", 13), fg_color="#16228E", text_color="white",
            corner_radius=8, width=90, height=32, command=self.next_page
        )
        self.next_button.place(relx=0.4, rely=0.5, anchor="center")

        self.view_result_button = ctk.CTkButton(
            self.bottom_frame, text="Buka Hasil Diagnosis",
            font=("Poppins Bold", 13), fg_color="#2ECC71", text_color="white",
            hover_color="#27ae60",
            corner_radius=8, width=180, height=32, command=self.open_selected_result
        )
        self.view_result_button.place(relx=0.85, rely=0.5, anchor="center")

        # Event double-click
        self.tree.bind("<Double-1>", self.open_diagnosis_result)

        # Pagination State
        self.page = 1
        self.items_per_page = 10
        self.total_pages = 1
        self.data_list = []
        self.filtered_data = []

        self.load_history_data()

    def on_show(self):
        """Dipanggil otomatis ketika layar riwayat ditampilkan untuk memuat data terbaru."""
        self.search_var.set("")
        self.load_history_data()

    def load_history_data(self):
        """Mengambil data riwayat pasien langsung dari database SQLite."""
        self.data_list = []
        
        try:
            if DATABASE_PATH.exists():
                conn = sqlite3.connect(DATABASE_PATH)
                c = conn.cursor()
                c.execute("SELECT id, nama, tanggal_pemeriksaan, path_gambar, path_segmentasi FROM pasien ORDER BY id DESC")
                rows = c.fetchall()
                conn.close()

                for idx, row in enumerate(rows, start=1):
                    p_id, nama, tgl, p_img, p_seg = row
                    status = "Tersedia" if (p_seg and Path(p_seg).exists()) else ("Gambar Ada" if (p_img and Path(p_img).exists()) else "Belum Selesai")
                    self.data_list.append({
                        "no": idx,
                        "id": p_id,
                        "nama": nama or "-",
                        "tanggal": tgl or "-",
                        "status": status,
                        "path_gambar": p_img,
                        "path_segmentasi": p_seg
                    })
        except Exception as e:
            print(f"Error reading database in history screen: {e}")

        # Fallback ke folder jika database belum memiliki data
        if not self.data_list and PATIENT_DATA_PATH.exists():
            folder_list = [f for f in PATIENT_DATA_PATH.iterdir() if f.is_dir()]
            for idx, folder in enumerate(folder_list, start=1):
                parts = folder.name.split('_')
                tgl = parts[0] if len(parts) > 0 else "-"
                nama = parts[1] if len(parts) > 1 else folder.name
                seg_path = folder / "output-segmented.jpg"
                status = "Tersedia" if seg_path.exists() else "Belum Selesai"
                self.data_list.append({
                    "no": idx,
                    "id": idx,
                    "nama": nama,
                    "tanggal": tgl,
                    "status": status,
                    "path_gambar": str(folder / "captured-image-original.png"),
                    "path_segmentasi": str(seg_path) if seg_path.exists() else None
                })

        self.filtered_data = self.data_list
        self.total_pages = max(1, (len(self.filtered_data) + self.items_per_page - 1) // self.items_per_page)
        self.page = 1
        self.update_table()

    def search_patient(self, event=None):
        """Filter daftar riwayat berdasarkan kata kunci nama pasien."""
        search_query = self.search_var.get().strip().lower()

        if not search_query:
            self.filtered_data = self.data_list
        else:
            self.filtered_data = [
                row for row in self.data_list if search_query in str(row["nama"]).lower()
            ]

        self.total_pages = max(1, (len(self.filtered_data) + self.items_per_page - 1) // self.items_per_page)
        self.page = 1
        self.update_table()

    def update_table(self):
        """Render isi tabel sesuai halaman yang aktif."""
        self.tree.delete(*self.tree.get_children())

        start = (self.page - 1) * self.items_per_page
        end = start + self.items_per_page
        for row in self.filtered_data[start:end]:
            self.tree.insert("", "end", iid=str(row["id"]), values=(
                row["no"],
                row["id"],
                row["nama"],
                row["tanggal"],
                row["status"]
            ))

        self.page_label.configure(text=f"Page {self.page} of {self.total_pages}")
        self.prev_button.configure(state="normal" if self.page > 1 else "disabled")
        self.next_button.configure(state="normal" if self.page < self.total_pages else "disabled")

    def open_selected_result(self):
        """Buka hasil diagnosis untuk baris yang sedang dipilih."""
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showinfo("Pilih Pasien", "Silakan pilih salah satu data pasien dari tabel terlebih dahulu.")
            return
        self.open_record_by_id(selected_item[0])

    def open_diagnosis_result(self, event):
        """Event saat baris tabel di-double click."""
        selected_item = self.tree.selection()
        if not selected_item:
            return
        self.open_record_by_id(selected_item[0])

    def open_record_by_id(self, item_id):
        """Menyiapkan data pasien dan membuka halaman DiagnosisResultScreen."""
        matched = next((item for item in self.data_list if str(item["id"]) == str(item_id)), None)
        if not matched:
            return

        self.controller.current_patient_id = matched["id"]
        self.controller.current_patient_nama = matched["nama"]
        self.controller.current_patient_tanggal = matched["tanggal"]
        self.controller.segmentation_output_path = matched["path_segmentasi"]
        self.controller.current_result_data = matched

        self.controller.show_frame("DiagnosisResultScreen")

    def prev_page(self):
        if self.page > 1:
            self.page -= 1
            self.update_table()

    def next_page(self):
        if self.page < self.total_pages:
            self.page += 1
            self.update_table()
