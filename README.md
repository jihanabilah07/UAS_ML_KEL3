# Rock Paper Scissors Image Classifier

## Deskripsi Proyek
Aplikasi ini adalah sistem klasifikasi gambar berbasis Convolutional Neural Network (CNN) yang dapat mengenali gestur tangan rock (batu), paper (kertas), dan scissors (gunting) secara real-time. Aplikasi ini dibuat menggunakan framework TensorFlow untuk pemodelan dan Streamlit untuk antarmuka pengguna.

### Fitur Utama:
- Klasifikasi gambar rock, paper, scissors melalui upload file
- Pengambilan gambar melalui kamera perangkat
- Deteksi gestur tangan secara real-time
- Penanganan gambar non-RPS (bukan rock, paper, scissors)
- Visualisasi probabilitas prediksi
- Antarmuka pengguna yang intuitif dan responsif

## Anggota Tim
1. Jihan Nabilah - 2208107010035 (Ketua)
2. Shofia Nurul Huda - 2208107010015
3. Farhanul Khair - 2208107010076
4. M. Bintang Indra Hidayat - 2208107010023
5. Ahmad Syah Ramadhan - 2208107010033

**Kelas:** A 
**Mata Kuliah:** Praktikum Pembelajaran Mesin A

## Demo
![Demo Aplikasi](assets/demo.gif)

## Teknologi yang Digunakan
- Python 3.8+
- TensorFlow 2.x
- Streamlit
- OpenCV
- Pandas
- NumPy
- Pillow (PIL)
- streamlit-webrtc

## Instalasi dan Penggunaan

### Prasyarat
- Python 3.8 atau lebih baru
- Pip (Python package manager)

### Langkah Instalasi

1. Clone repositori ini:
```bash
git clone hhttps://github.com/jihanabilah07/UAS_ML_KEL3
cd rock-paper-scissors-classifier
```

2. Buat dan aktifkan virtual environment:
```bash
# Untuk Windows
python -m venv venv
venv\Scripts\activate

# Untuk MacOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Jalankan aplikasi:
```bash
streamlit run app.py
```

5. Buka browser dan akses aplikasi di:
```
http://localhost:8501
```

### Struktur Direktori
```
rock-paper-scissors-classifier/
├── app.py                      # Aplikasi utama Streamlit
├── models/                     # Direktori untuk model
│   └── best_model.h5           # Model CNN terlatih
├── class_mapping.csv           # Pemetaan indeks kelas ke nama kelas
├── requirements.txt            # Dependencies
├── assets/                     # Gambar dan aset lainnya
└── README.md                   # Dokumentasi proyek
```

## Penggunaan Aplikasi

### Upload Image
1. Pilih tab "Upload Image" pada sidebar
2. Klik "Choose an image..." dan pilih file gambar
3. Aplikasi akan menampilkan gambar yang dipilih dan hasil klasifikasi

### Take Photo
1. Pilih tab "Take Photo" pada sidebar
2. Izinkan akses kamera pada browser
3. Klik tombol untuk mengambil foto
4. Aplikasi akan menampilkan gambar yang diambil dan hasil klasifikasi

### Real-time Detection
1. Pilih tab "Real-time Camera Detection" pada sidebar
2. Izinkan akses kamera pada browser
3. Tunjukkan gestur tangan (rock, paper, scissors) di depan kamera
4. Aplikasi akan menampilkan prediksi secara real-time

## Model
Model yang digunakan adalah Convolutional Neural Network (CNN) yang dilatih menggunakan dataset Rock-Paper-Scissors. Arsitektur model terdiri dari beberapa lapisan konvolusi dengan fungsi aktivasi ReLU, max pooling, dan lapisan fully connected. Model ini mencapai akurasi sekitar 97% pada dataset validasi.

## Troubleshooting
- **Kamera tidak berfungsi**: Pastikan browser memiliki izin untuk mengakses kamera dan tidak ada aplikasi lain yang menggunakan kamera
- **Prediksi tidak akurat**: Pastikan gestur tangan ditunjukkan dengan jelas di depan kamera dengan pencahayaan yang cukup
- **Error saat menjalankan aplikasi**: Pastikan semua dependencies terinstal dengan benar dan versi yang sesuai

## Kontribusi
Jika Anda ingin berkontribusi pada proyek ini, silakan fork repositori, buat branch fitur baru, dan kirim pull request.

## Link GitHub Repository
[GitHub Repository](https://github.com/jihanabilah07/UAS_ML_KEL3)
