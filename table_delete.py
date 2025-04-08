import sqlite3

# Koneksi ke database pasien.db
conn = sqlite3.connect('pasien.db')
c = conn.cursor()

# Menghapus semua data dalam tabel
c.execute("DELETE FROM pasien")

# Commit perubahan dan tutup koneksi
conn.commit()
conn.close()

print("Semua data dalam tabel 'pasien' telah dihapus.")
