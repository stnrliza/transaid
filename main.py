# I added comments in every files for explanations to help understanding the code easier, hope it helps! (actually it's for me too, hehe)
import tkinter as tk
import ttkbootstrap as tb

# Mengimpor semua kelas frame dari file yang relevan
from a_welcome_screen import TransAIDScreen
from b1_patient_data import PatientDataScreen
from b2_diagnosis_history import DiagnosisHistoryScreen
from c_live_camera import LiveCameraScreen
from d_loading_screen import LoadingScreen
from e_diagnosis_result import DiagnosisResultScreen
from start_push_button import PushButtonReader  # Import kelas untuk push button

class MainApp(tb.Window):
    def __init__(self):
        super().__init__(themename="flatly")
        # Mengatur jendela aplikasi utama
        self.title("TransAID")
        self.geometry("1920x1080")  # Mengatur ukuran jendela agar lebih besar
        self.state('zoomed')  # Mengatur window menjadi full screen

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

        self.frames[frame_name].pack(fill="both", expand=True)
        self.frames[frame_name].tkraise()

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()