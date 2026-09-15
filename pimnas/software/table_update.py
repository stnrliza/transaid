import sqlite3

# Koneksi ke database pasien.db
conn = sqlite3.connect('pasien.db')
c = conn.cursor()

# Menambahkan kolom path_gambar ke tabel pasien jika belum ada
try:
    c.execute("ALTER TABLE pasien ADD COLUMN path_gambar TEXT")
    print("Kolom 'path_gambar' berhasil ditambahkan.")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e):
        print("Kolom 'path_gambar' sudah ada.")
    else:
        print(f"Error saat menambahkan kolom 'path_gambar': {e}")

# Menambahkan kolom path_segmentasi ke tabel pasien jika belum ada
try:
    c.execute("ALTER TABLE pasien ADD COLUMN path_segmentasi TEXT")
    print("Kolom 'path_segmentasi' berhasil ditambahkan.")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e):
        print("Kolom 'path_segmentasi' sudah ada.")
    else:
        print(f"Error saat menambahkan kolom 'path_segmentasi': {e}")

# Menutup koneksi ke database
conn.commit()
conn.close()
