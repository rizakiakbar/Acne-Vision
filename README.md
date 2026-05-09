🧠 Deteksi Emosi Pengguna dari Teks Berbahasa Indonesia

Dokumentasi Project — Bidirectional LSTM · Flask API · NLP Bahasa Indonesia


1. Gambaran Umum
Project ini adalah sistem yang bisa mendeteksi apakah sebuah teks berbahasa Indonesia mengandung emosi Positif atau Negatif. Idenya sederhana — pengguna kirim teks, sistem kasih tahu sentimennya.
Di baliknya, kita pakai model Bidirectional LSTM yang dilatih khusus dengan data Bahasa Indonesia, jadi model ini ngerti konteks dan nuansa bahasa lokal, termasuk slang. Model disimpan dalam format .h5 dan diakses lewat API Flask.
📌 InputTeks bebas Bahasa Indonesia📌 OutputLabel (Positif / Negatif) + Confidence Score📌 ModelBidirectional LSTM — akurasi validasi ~83%📌 APIFlask (REST API)

2. Teknologi yang Digunakan
Bahasa & Framework
KomponenTeknologi / LibraryKeteranganBahasaPythonMain languageDeep LearningTensorFlow, KerasBangun & latih model LSTMML Utilitiesscikit-learnClass weight, label encoder, dll.Data HandlingNumPy, PandasOlah data numerik & tabelNLPNLTK, rePreprocessing teksDeploymentFlaskREST API untuk inferensi model
File Output Model
FileFungsimodel_saya.h5Model Bidirectional LSTM yang sudah dilatihtokenizer.jsonTokenizer yang dipakai saat training — harus dipakai juga saat inferensi

3. Alur Machine Learning
3.1 Preprocessing Data
Sebelum data masuk ke model, ada beberapa tahap pembersihan yang harus dilalui:

Konversi semua teks ke huruf kecil (lowercase)
Hapus tanda baca, angka, dan karakter spesial pakai regex
Konversi slang word Bahasa Indonesia ke bentuk baku
Tokenisasi teks dan padding ke panjang maksimal 30 token
Encoding label teks menjadi angka pakai LabelEncoder dari sklearn
Pembagian data: 80% untuk training, 20% untuk testing

3.2 Penanganan Class Imbalance
Kalau jumlah data tiap kelas tidak seimbang (misalnya data Negatif jauh lebih banyak dari Positif), model bisa jadi bias. Solusinya, kita pakai class_weight='balanced' dari sklearn supaya setiap kelas dapet bobot yang proporsional saat training.

4. Arsitektur Model
Model dibangun pakai pendekatan Bidirectional LSTM — artinya model membaca urutan teks dari kiri ke kanan dan kanan ke kiri secara bersamaan. Ini bikin model lebih ngerti konteks kalimat.
Kode Arsitektur
pythonmodel = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=30))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
Penjelasan Tiap Layer
LayerOutput ShapeFungsiEmbedding(None, 30, 128)Ubah token jadi vektor dense 128 dimensiSpatialDropout1D(0.2)(None, 30, 128)Regularisasi — cegah overfitting di level embeddingBidirectional LSTM(64)(None, 30, 128)Baca konteks dari dua arahGlobalMaxPooling1D(None, 128)Ambil fitur paling penting dari sequenceDense(64, relu)(None, 64)Layer fully-connected untuk representasi lebih tinggiDropout(0.5)(None, 64)Regularisasi tambahan sebelum outputDense(2, softmax)(None, 2)Output: probabilitas kelas Positif dan Negatif
Konfigurasi Training
ParameterNilaiLoss Functionsparse_categorical_crossentropyOptimizerAdamMetricsAccuracyEarly Stoppingmonitor='val_loss', patience=3, restore_best_weights=TrueReduceLROnPlateaumonitor='val_loss', factor=0.2, patience=2

5. Hasil Training

✅ Akurasi Validasi Akhir : ~83%
📦 Model tersimpan di : model/model_saya.h5
🔤 Tokenizer tersimpan di : model/tokenizer.json

Early stopping aktif — training berhenti otomatis kalau val_loss tidak membaik selama 3 epoch berturut-turut, dan bobot terbaik langsung dipulihkan. Ini cara yang baik buat hindari overfitting tanpa harus manual ngecek terus.

6. Prediksi (Inferensi)
Setelah model dilatih, ini fungsi yang dipakai untuk prediksi teks baru:
pythondef predict_text(text, model, tokenizer, label_encoder, maxlen=30):
    sequence = tokenizer.texts_to_sequences([text])
    padded   = pad_sequences(sequence, maxlen=maxlen)
    prediction      = model.predict(padded)
    predicted_index = np.argmax(prediction)
    label      = label_encoder.inverse_transform([predicted_index])[0]
    confidence = prediction[0][predicted_index]
    return label, confidence
Contoh Input & Output
Input TeksLabelConfidence"Aku merasa sangat sedih dan tidak punya semangat."Negative91.2%"Hari ini menyenangkan banget, semua berjalan lancar!"Positive88.7%

7. Struktur File Project
project/
├── model/
│   ├── model_saya.h5       ← model LSTM hasil training
│   └── tokenizer.json      ← tokenizer yang dipake saat training
├── notebook/
│   └── training.ipynb      ← proses EDA, preprocessing, training
├── api/
│   └── app.py              ← Flask REST API
├── data/
│   └── dataset.csv         ← dataset teks Bahasa Indonesia
└── requirements.txt        ← daftar dependensi

8. Rencana Pengembangan Selanjutnya
Project ini masih bisa dikembangkan lebih jauh. Beberapa ide yang menarik untuk dieksplor:

Multi-label emosi — bukan cuma Positif/Negatif, tapi bisa deteksi marah, senang, takut, sedih, kaget secara terpisah
Analisis tren emosi harian — simpan histori prediksi dan tampilkan grafik perubahan emosi dari waktu ke waktu
Integrasi webhook — sambungkan ke chatbot, WhatsApp API, atau Telegram bot
Fine-tuning dengan IndoBERT — upgrade model ke transformer berbasis bahasa Indonesia untuk akurasi lebih tinggi
Deploy ke cloud — hosting di Heroku, Railway, atau GCP Cloud Run supaya bisa diakses publik


9. Lisensi & Kontribusi
Project ini open source dan bebas dipakai untuk keperluan belajar maupun eksperimen. Kalau mau kontribusi — baik itu tambah fitur, perbaiki bug, atau improve dokumentasi — sangat terbuka. Fork dulu, buat branch baru, terus pull request.
