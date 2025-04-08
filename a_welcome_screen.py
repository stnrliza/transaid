import ctypes
import tkinter as tk
import customtkinter as ctk
from pathlib import Path
from PIL import Image, ImageTk

ASSETS_PATH = Path(__file__).parent / "assets" / "frame-a"  # needing .parent bcs if not, it's as if "assets" folder inside this file; "a.py" --well, that's totally wrong, right?

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / path  # ASSETS_PATH is used to address just the folder ("assets" folder), while this function is used to *address each assets inside frame-a* (e.g. button, image, etc)

# activing DPI Awareness (shortly it's for adjusting the screen to high resolution/high DPI monitor)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception as e:
    print(f"Error setting DPI awareness: {e}")

class TransAIDScreen(tk.Frame): 
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#FFFFFF") 

        # frame for filling elements (logo picture, title, button, etc)
        self.container = tk.Frame(self, bg="#FFFFFF") 
        self.container.place(relx=0, rely=0, relwidth=1, relheight=1) # relwidth & relheight explained below in the docstring

        # updating the screen size for changing in size
        self.update_idletasks() # making sure the size is updated before shown
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        """
        === PADDING CONFIGURATIONS ===
        there are 3 main layout methods in Tkinter: grid(), pack(), and place(). Here's the difference:
        1. grid() <-- uses a table-like structure (row and column)
        - padx=(left, right)                              : space between columns (|)  
        - pady=(top, bottom)                              : space between rows (=)  
        2. pack() <-- uses a stacking structure (top-down left-right)  
        - padx=(left, right)                              : space between the asset and its parent (frame named container, in this context)  
        - pady=(top, bottom)                              : space between the asset and its parent  
        3. place() <-- positioning element, (0, 0) starts from left top as default
           - BASED ON POSITION
             a. absolute coordinate (x=..., y=...)          : position of the element based on precise coordinate
             b. relative coordinate (relx=..., rely=...)    : position of the element relative to the parent
           TYPE OF ANCHOR FOR POSITIONING
           3.1. (anchor="n") or north
           3.2. (anchor="s") or south
           3.3. (anchor="e") or east
           3.4. (anchor="w") or west
           3.5. (anchor="ne") or north east
           3.6. (anchor="nw") or north west
           3.7. (anchor="se") or south east
           3.8. (anchor="sw") or south west
           3.9. (anchor="center")
           *note: the anchor relative point is the same as the name. e.g. west <-- left, center ; south <-- center, bottom ; north east <-- right, top
           - BASED ON SIZE
             a. absolute size (width=..., length=...)       : precise coordinate of the size in pixel
             b. relative size (relwidth=..., relheight=...) : size relative to the parent's
           *note for relative:
            - relative (both position and size) allows the program to be responsive and adaptive to the resizing window
            - must be between 0 - 1 (e.g. 0.01, 0.5, 0.799, etc)
        
        overall conclusion:
        in one same container, we are able to choose JUST ONE CONFIGURATION. if not, it'll be crashing (e.g. using pack and grid in the same container <-- error)
        """

        # adding asset --> logo
        logo_path = relative_to_assets("a-image.png")  #so yea, we get the path of a specific asset, which is a-image.png in here, by using the "relative_to_assets" function mentioned earlier
        if logo_path.exists():
            logo_image = Image.open(logo_path)
            responsive_logo_size = logo_image.resize((int(screen_width*0.225), int(screen_height*0.4)), Image.LANCZOS) # maintaining the default size 432x432, and Image.LANCZOS is used for keeping the resized image sharp
            logo = ImageTk.PhotoImage(responsive_logo_size)
            self.logo_label = tk.Label(self.container, image=logo, bg="#FFFFFF")
            self.logo_label.image = logo # saving the reference (address of the logo asset) bcs if not, Tkinter will detele it thinking it's no longer in use
            self.logo_label.place(relx=0.5, rely=0.1, anchor="n")  # explained above in the docstring
        else:
            print(f"Error: File tidak ditemukan - {logo_path}")
            
        # writing the title "TransAID" in screen (btw, bg used in tkinter, fg_color usesd in customtkinter; both are used for background color)
        self.font_size = int(screen_height / 18)
        self.title_label = ctk.CTkLabel(
            self.container,
            text="TransAID",
            font=("Poppins Bold", self.font_size),
            text_color="#16228E",
            fg_color="transparent")
        self.title_label.place(relx=0.5, rely=0.6, anchor="n")

        # making functional ''enroll'' button; so if clicked, it will navigate to "b1_patient_data.py" screen
        self.button_font_size = int(self.font_size / 3)
        self.daftar_button = ctk.CTkButton(
            self.container,
            text="Daftar",
            font=("Poppins Medium", self.button_font_size),
            fg_color="#A8DEE6",
            text_color="#16228E",
            corner_radius=15,
            command=lambda: controller.show_frame("PatientDataScreen")
        )
        self.daftar_button.place(relx=0.3, rely=0.8, anchor="center", relwidth=0.2, relheight=0.1)

        # making functional ''see patients' history result'' button; so if clicked, it will navigate to "b2_diagnosis_history.py" screen
        self.riwayat_button = ctk.CTkButton(
            self.container,
            text="Lihat Riwayat",
            font=("Poppins Medium", self.button_font_size),
            fg_color="#A8DEE6", 
            text_color="#16228E", 
            corner_radius=15,
            command=lambda: controller.show_frame("DiagnosisHistoryScreen")
        )
        self.riwayat_button.place(relx=0.7, rely=0.8, anchor="center", relwidth=0.2, relheight=0.1)

        self.bind("<Configure>", self.on_resize) # so that all elements' positions and sizes always updated when screen (frame) is resized

    def on_resize(self, event):
        """Update elements' size and position when window size changes"""
        new_font_size = int(event.height / 18)
        self.title_label.configure(font=("Poppins Bold", new_font_size))
        self.daftar_button.configure(font=("Poppins Medium", int(new_font_size / 3)))
        self.riwayat_button.configure(font=("Poppins Medium", int(new_font_size / 3)))