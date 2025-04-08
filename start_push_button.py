import serial
import threading

class PushButtonReader:
    def __init__(self, port='COM9', baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.connect_serial()

    def connect_serial(self):
        """Mencoba menghubungkan ke port serial dan menangani error"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"Connected to {self.port} at {self.baudrate} baud.")
        except serial.SerialException as e:
            print(f"Failed to connect to {self.port}: {e}")
            self.ser = None

    def read_push_button_status(self):
        """Membaca status push button dari port serial"""
        if self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line == 'SHORT_PRESS':
                    return "short_press"
                elif line == 'LONG_PRESS':
                    return "long_press"
                return None
            except serial.SerialException as e:
                print(f"Error reading from {self.port}: {e}")
                self.close_connection()
                return None
        else:
            print("Serial connection not open")
            return None

    def continuously_monitor(self, callback, interval=100):
        """Monitor status push button secara terus-menerus"""
        def monitor():
            while True:
                status = self.read_push_button_status()
                if status == 'short_press':
                    callback()  # Panggil callback jika short_press terdeteksi
                threading.Event().wait(interval / 1000.0)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def close_connection(self):
        """Menutup koneksi serial"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed.")
