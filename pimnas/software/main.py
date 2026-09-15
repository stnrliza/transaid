# I added comments in every files for explanations to help understanding the code easier, hope it helps! (actually it's for me too, hehe)
import ctypes
import tkinter as tk
import customtkinter as ctk

# Mengaktifkan DPI Awareness (satu kali saja, di file utama)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

# Mengimpor semua kelas frame dari file yang relevan
from a_welcome_screen import TransAIDScreen
from b1_patient_data import PatientDataScreen
from b2_diagnosis_history import DiagnosisHistoryScreen
from c_live_camera import LiveCameraScreen
from d_loading_screen import LoadingScreen
from e_diagnosis_result import DiagnosisResultScreen

class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        # Mengatur jendela aplikasi utama
        self.title("TransAID")
        self.geometry("1920x1080")  # Mengatur ukuran jendela agar lebih besar
        self.state('zoomed')  # Mengatur window menjadi full screen

        # Shared state untuk komunikasi antar screen
        self.current_patient_folder = None    # Folder pasien yang sedang aktif
        self.current_result_data = None       # Data hasil diagnosis untuk ditampilkan
        self.segmentation_output_path = None  # Path output segmentasi terbaru

        # Container untuk menampung semua frame
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        # Dictionary untuk menampung semua frames
        self.frames = {}

        # Inisialisasi semua frames dan tambahkan ke dictionary menggunakan nama string
        for F, name in [
            (TransAIDScreen, "TransAIDScreen"),
            (PatientDataScreen, "PatientDataScreen"),
            (DiagnosisHistoryScreen, "DiagnosisHistoryScreen"),
            (LiveCameraScreen, "LiveCameraScreen"),
            (LoadingScreen, "LoadingScreen"),
            (DiagnosisResultScreen, "DiagnosisResultScreen")
        ]:
            frame = F(self.container, self)
            self.frames[name] = frame
            frame.pack(fill="both", expand=True)

        # Tampilkan frame awal
        self.show_frame("TransAIDScreen")

    def show_frame(self, frame_name):
        """Bring the frame to the front for display"""
        if frame_name not in self.frames:
            if frame_name == "PatientDataScreen":
                self.frames[frame_name] = PatientDataScreen(self.container, self)
            elif frame_name == "DiagnosisHistoryScreen":
                self.frames[frame_name] = DiagnosisHistoryScreen(self.container, self)
            self.frames[frame_name].pack(fill="both", expand=True)

        # hiding all frames before showing the next
        for frame in self.frames.values():
            frame.pack_forget()

        frame = self.frames[frame_name]
        frame.pack(fill="both", expand=True)
        frame.tkraise()

        # Panggil on_show() jika frame mengimplementasikannya
        # (untuk refresh data, re-init kamera, dll)
        if hasattr(frame, 'on_show'):
            frame.on_show()

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
