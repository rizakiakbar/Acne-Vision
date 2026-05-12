# 🧴 Acne Detection
Website untuk mendeteksi tingkat keparahan jerawat dari gambar menggunakan teknologi Machine Learning dan Computer Vision.

---

## 🚀 Fitur

- **Upload Gambar**: Kirim gambar wajah untuk dianalisis
- **Deteksi Level Jerawat**: Klasifikasi otomatis ke 4 tingkat keparahan (Level 0–3)
- **Confidence Score**: Persentase keyakinan model terhadap hasil prediksi
- **Probabilitas Tiap Kelas**: Tampilkan distribusi probabilitas untuk semua level
- **Responsive Design**: Tampilan optimal di desktop dan mobile
- **Error Handling**: Validasi file dan penanganan error yang komprehensif

---

## 🛠️ Setup & Installation

1. Clone atau download project ini
2. Buka folder project di VS Code
3. Buka terminal dan jalankan `pip install -r requirements.txt`
4. Jalankan API dengan `python app.py`
5. Buka browser dan akses `http://localhost:5000`

> **Catatan**: Pastikan file `best_model.h5` sudah ada di folder sebelum menjalankan server.

---

## 📂 Struktur File

```
acne-detection/
├── Classification/
│   ├── JPEGImages/          ← folder dataset gambar
│   ├── NNEW_trainval_*.txt  ← file list data training
│   └── NNEW_test_*.txt      ← file list data testing
├── output/
│   ├── acne_cnn_final.h5        ← model CNN final
│   ├── training_history.png     ← grafik akurasi & loss
│   └── confusion_matrix.png     ← confusion matrix
├── best_model.h5            ← model terbaik (dipakai API)
├── app.py                   ← Flask API
├── notebook.ipynb           ← proses training model
└── requirements.txt         ← daftar dependensi
```

---

## 📝 Format API Response

API mengembalikan response dalam format JSON:

```json
{
  "predicted_class": "Level 1",
  "confidence": "87.3%",
  "probabilities": {
    "Level 0": "5.1%",
    "Level 1": "87.3%",
    "Level 2": "6.2%",
    "Level 3": "1.4%"
  }
}
```

### Deskripsi Field

| Field | Tipe | Keterangan |
|---|---|---|
| `predicted_class` | String | Hasil prediksi: `"Level 0"`, `"Level 1"`, `"Level 2"`, `"Level 3"` |
| `confidence` | String | Persentase keyakinan model terhadap kelas yang diprediksi |
| `probabilities` | Object | Distribusi probabilitas untuk semua 4 level jerawat |

### Keterangan Level

| Level | Keterangan |
|---|---|
| Level 0 | Tidak ada jerawat / sangat ringan (1–5 jerawat) |
| Level 1 | Ringan (6–20 jerawat) |
| Level 2 | Sedang (21–42 jerawat) |
| Level 3 | Parah (51–65+ jerawat) |

---

## 🧠 Tentang Model

Model yang digunakan adalah **CNN (Convolutional Neural Network)** kustom dengan 4 blok konvolusi, dilatih menggunakan dataset **NNEW Acne Level Dataset**.

### Arsitektur Singkat

```
Input (128×128×3)
│
├─ Conv Block 1: Conv2D(32) × 2 → BatchNorm → MaxPool → Dropout
├─ Conv Block 2: Conv2D(64) × 2 → BatchNorm → MaxPool → Dropout
├─ Conv Block 3: Conv2D(128) × 2 → BatchNorm → MaxPool → Dropout
├─ Conv Block 4: Conv2D(256) → BatchNorm → MaxPool → Dropout
│
├─ GlobalAveragePooling2D
├─ Dense(256) → BatchNorm → Dropout
├─ Dense(128) → Dropout
└─ Dense(4, softmax) ← Output
```

### Hasil Training

| Metrik | Nilai |
|---|---|
| Test Accuracy | **93.22%** |
| Best Val Accuracy | **94.25%** |
| Test Loss | 0.2435 |
| Total Epoch | 50 |
| Total Parameter | 685,220 |

### Performa per Kelas

| Kelas | Precision | Recall | F1-Score |
|---|---|---|---|
| Level 0 | 0.95 | 0.97 | 0.96 |
| Level 1 | 0.97 | 0.91 | 0.93 |
| Level 2 | 0.79 | 0.91 | 0.85 |
| Level 3 | 0.92 | 0.95 | 0.93 |

---

## ⚙️ Konfigurasi Model

Kalau mau ubah parameter training, edit bagian `CONFIG` di notebook:

```python
CONFIG = {
    "img_size"      : (128, 128),   # Ukuran input gambar
    "batch_size"    : 32,
    "epochs"        : 50,
    "num_classes"   : 4,
    "learning_rate" : 1e-3,
    "data_dir"      : "Classification/JPEGImages",
}
```

---

## 🔧 Error Handling

API sudah dilengkapi penanganan error untuk:

- Format file tidak didukung (bukan `.jpg` / `.png`)
- Ukuran file terlalu besar
- Gambar gagal dibaca atau corrupt
- Model belum dimuat
- Error server internal (500)

---

## 📱 Browser Support

- ✅ Chrome (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge

---

## 📋 TODO / Future Enhancements

- Tambah rekomendasi skincare berdasarkan level jerawat
- Deteksi area wajah secara otomatis sebelum analisis
- Riwayat analisis per pengguna
- Export hasil ke PDF
- Upgrade backbone ke MobileNetV2 atau EfficientNet untuk performa lebih baik
- Deploy ke cloud (Heroku / Railway / GCP)

---

## 🤝 Contributing

1. Fork repository ini
2. Buat branch baru (`git checkout -b feature/NamaFitur`)
3. Commit changes (`git commit -m 'Add NamaFitur'`)
4. Push ke branch (`git push origin feature/NamaFitur`)
5. Buat Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📞 Contact

Kalau ada pertanyaan atau issues, silakan buat issue di repository ini.

Happy Coding! 🎉
