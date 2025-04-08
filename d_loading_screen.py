import ctypes
import os
import ttkbootstrap as tb
import threading
from pathlib import Path
import tkinter as tk
from tkinter import Canvas, PhotoImage
import time
from yolov8_segment import run_yolov8_segmentation

ASSETS_PATH = Path(__file__).resolve().parent / "assets" / "frame-d"

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Mengaktifkan DPI Awareness
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception as e:
    print(f"Error setting DPI awareness: {e}")

class LoadingScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#FFFFFF")

        # Header for the loading screen
        header_frame = tk.Frame(self, bg="#FFFFFF")
        header_frame.pack(fill="both", pady=(20, 0))

        # Membuat canvas untuk gambar latar belakang
        canvas = Canvas(header_frame, bg="#FFFFFF", height=400, width=1920, bd=0, highlightthickness=0, relief="ridge")
        canvas.pack(fill="x")
        image_image_1_path = relative_to_assets("d-image.png")
        if image_image_1_path.exists():
            image_image_1 = PhotoImage(file=image_image_1_path)
            canvas.create_image(959, 200, image=image_image_1)
            canvas.image_image_1 = image_image_1
        else:
            print(f"Error: File not found - {image_image_1_path}")

        # Text to display processing message
        message_label = tk.Label(self, text="Sedang Memproses", font=("Poppins Bold", 50), fg="#15218E", bg="#FFFFFF")
        message_label.pack(pady=50)

        # Floodgauge (loading bar)
        style = tb.Style()
        style.configure("primary.Horizontal.TFloodgauge",
                        background="#15218E",
                        troughcolor="#DDDDDD",
                        lightcolor="#15218E",
                        darkcolor="#15218E",
                        bordercolor="#15218E")

        self.my_gauge = tb.Floodgauge(self, bootstyle="primary", font=("Helvetica", 18), mask="{}%",
                                      maximum=100, orient="horizontal", value=0, mode="determinate")
        self.my_gauge.pack(pady=(20, 0), fill="x", padx=150)

    def start_segmentation(self, image_path, patient_folder):
        """
        Starts segmentation and progress bar in parallel using threads.
        `image_path`: Path to the captured image.
        `patient_folder`: Folder where both the captured image and the segmented output will be saved.
        """
        self.image_path = image_path
        self.patient_folder = patient_folder  # Patient folder from the input in bbb1.py

        # Start two threads: one for progress bar and one for segmentation
        threading.Thread(target=self.update_progress_bar, daemon=True).start()
        threading.Thread(target=self.run_segmentation, daemon=True).start()

    def update_progress_bar(self):
        """Updates the progress bar as the segmentation happens."""
        for i in range(101):
            self.my_gauge.configure(value=i)
            self.update_idletasks()
            time.sleep(0.05)  # Delay to simulate progress

    def run_segmentation(self):
        """Runs the YOLOv8 segmentation in a separate thread."""
        output_path = os.path.join(self.patient_folder, "output-segmented.jpg")
        run_yolov8_segmentation(self.image_path, output_path)
        
        # Once segmentation is complete, navigate to the results screen
        self.controller.show_frame("DiagnosisResultScreen")
        self.controller.frames["DiagnosisResultScreen"].load_segmented_image(output_path)