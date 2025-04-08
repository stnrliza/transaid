from ultralytics import YOLO
import cv2
import numpy as np
import os
import sqlite3

# Load YOLOv8 model
model = YOLO('best.pt')

# Fungsi untuk menjalankan segmentasi YOLOv8 dan menyimpan hasil
def run_yolov8_segmentation(image_path, output_path):
    # Load the image
    print(f"Attempting to load image for segmentation: {image_path}")
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is None:
        raise ValueError(f"Error: Image not found or cannot be read at {image_path}")

    print(f"Image loaded successfully from {image_path}. Starting segmentation...")

    # Jalankan prediksi pada gambar
    results = model(image)

    # Cek hasil prediksi
    print(f"Results from YOLOv8: {results}")
    
    # Siapkan gambar asli untuk overlay
    orig_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_overlay = np.zeros_like(orig_img, dtype=np.uint8)

    if len(results) == 0:
        print("No results from YOLOv8 segmentation, but forcing segmentation.")
    else:
        print("Processing results...")
    
    # Visualisasikan dan proses hasil
    for result in results:
        masks = result.masks.data.numpy() if result.masks else None

        if masks is not None:
            # Loop untuk menerapkan mask (karies) ke gambar
            for i, mask in enumerate(masks):
                mask_resized = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]))
                mask_uint8 = (mask_resized * 255).astype(np.uint8)

                # Ambil nilai confidence score dari deteksi
                confidence_score = result.boxes.conf[i] * 100  # Konversi ke persen

                # Berikan warna pada mask berdasarkan confidence score
                if 1 <= confidence_score < 50:
                    mask_overlay[:, :, 1] = np.maximum(mask_overlay[:, :, 1], mask_uint8)  # Hijau untuk 1-49%
                elif 50 <= confidence_score <= 100:
                    mask_overlay[:, :, 0] = np.maximum(mask_overlay[:, :, 0], mask_uint8)  # Merah untuk 50-100%

                # Tentukan posisi teks confidence
                text_position = (10, 40 * (i + 1))

                # Buat background hitam untuk teks
                text = f'{confidence_score:.2f}%'
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                top_left = (text_position[0] - 10, text_position[1] - text_height - 10)
                bottom_right = (text_position[0] + text_width + 10, text_position[1] + baseline + 5)
                cv2.rectangle(orig_img, top_left, bottom_right, (0, 0, 0), -1)

                # Tambahkan teks confidence ke gambar
                cv2.putText(orig_img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Jika tidak ada mask, tetap buat gambar output
    if np.count_nonzero(mask_overlay) == 0:
        print("No masks found, saving the original image without segmentation.")
    
    # Buat gambar dengan overlay yang diterapkan dengan transparansi
    alpha = 0.5
    segmented_img = cv2.addWeighted(orig_img, 1 - alpha, mask_overlay, alpha, 0)
    output_img_bgr = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR)

    # Simpan hasil segmentasi
    cv2.imwrite(output_path, output_img_bgr)
    print(f"Segmented output saved at: {output_path}")

    # Hitung persentase karies
    total_pixels = orig_img.shape[0] * orig_img.shape[1]
    caries_pixels = np.count_nonzero(mask_overlay[:, :, 0])  # Hitung jumlah piksel karies (merah)
    caries_percentage = caries_pixels / total_pixels
    print(f"Caries percentage: {caries_percentage * 100:.2f}%")

    # Simpan persentase karies ke dalam gambar
    cv2.putText(output_img_bgr, f'caries {caries_percentage * 100:.2f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Simpan gambar dengan persentase karies
    cv2.imwrite(output_path, output_img_bgr)
    print(f"Segmented output with caries percentage saved at: {output_path}")

# Fungsi untuk menyimpan path hasil segmentasi ke database
def save_segmented_path_to_db(path_segmentasi):
    # Koneksi ke database pasien.db
    conn = sqlite3.connect('pasien.db')
    c = conn.cursor()

    # Dapatkan ID pasien terakhir yang ditambahkan dan update dengan path segmentasi
    c.execute("SELECT id FROM pasien ORDER BY id DESC LIMIT 1")
    last_patient = c.fetchone()

    if last_patient:
        c.execute("UPDATE pasien SET path_segmentasi = ? WHERE id = ?", (path_segmentasi, last_patient[0]))
        conn.commit()
        print(f"Path hasil segmentasi berhasil disimpan ke database untuk pasien ID {last_patient[0]}.")

    conn.close()
