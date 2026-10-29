# AI Medis - Prediksi ICD-10 dengan ClinicalBERT

Aplikasi web berbasis Flask dan ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) untuk memprediksi kode diagnosis ICD-10 berdasarkan keluhan pasien berbahasa Indonesia.

## Fitur
- **Klasifikasi ICD-10**: Memprediksi diagnosis medis dari teks keluhan pasien.
- **Penerjemahan Otomatis**: Mendukung input bahasa Indonesia yang diterjemahkan secara otomatis sebelum diproses model.
- **Informasi ICD-10 Lengkap**: Menampilkan deskripsi diagnosis, catatan rubrik, dan eksklusi (exclusion) dari ICD-10 XML.
- **Analisis Entropi & Ketidakpastian**: Mengukur tingkat kepastian prediksi model (Rendah, Sedang, Tinggi).

## Struktur Proyek
- `app.py`: Aplikasi web utama Flask menggunakan model `model_clinicalbert_top10newww`.
- `main.py`: Versi alternatif menggunakan pipeline `AutoModelForSequenceClassification`.
- `templates/`: Template HTML untuk antarmuka web.
- `model/`:
  - `icd102019en.xml`: Data struktur XML ICD-10.
  - `dataset_translated.csv`: Mapping deskripsi kode ICD-10.
  - `model_clinicalbert_top10newww/`: Model PyTorch (`model.pth`), label mapping, dan konfigurasi tokenizer.

## Cara Menjalankan
1. Pasang dependensi:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan aplikasi web:
   ```bash
   python app.py
   ```
3. Buka browser di [http://127.0.0.1:5000](http://127.0.0.1:5000).
