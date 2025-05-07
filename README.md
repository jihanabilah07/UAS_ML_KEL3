# ✋🪨📄✂️ Rock Paper Scissors Image Classifier

## 📌 Deskripsi Proyek
Aplikasi ini merupakan sistem klasifikasi gambar berbasis **Convolutional Neural Network (CNN)** yang mampu mengenali gestur tangan **rock (batu)**, **paper (kertas)**, dan **scissors (gunting)** secara **real-time**. Dengan dukungan **TensorFlow** dan **Streamlit**, aplikasi ini menyediakan antarmuka yang ramah pengguna dan interaktif.

## 🚀 Fitur Utama
- 📁 Klasifikasi gambar melalui upload file
- 📸 Pengambilan gambar langsung dari kamera
- 🎥 Deteksi gestur tangan secara real-time
- ❌ Penanganan untuk gambar yang bukan RPS
- 📊 Visualisasi probabilitas prediksi
- 🖥️ Antarmuka pengguna intuitif & responsif

## 👨‍👩‍👧‍👦 Tim Pengembang

| Nama                           | NIM              | Peran        |
|-------------------------------|------------------|--------------|
| Jihan Nabilah                 | 2208107010035    | Ketua        |
| Shofia Nurul Huda             | 2208107010015    | Anggota      |
| Farhanul Khair                | 2208107010076    | Anggota      |
| M. Bintang Indra Hidayat      | 2208107010023    | Anggota      |
| Ahmad Syah Ramadhan           | 2208107010033    | Anggota      |

**Kelas:** A  
**Mata Kuliah:** Praktikum Pembelajaran Mesin A

## 🎥 Demo Aplikasi
![image](https://github.com/user-attachments/assets/c841658d-4a84-4514-8992-6334b18616dc)

## 🛠 Teknologi yang Digunakan
- 🐍 Python 3.8+
- 🧠 TensorFlow 2.x
- 🌐 Streamlit
- 🎦 OpenCV
- 📊 Pandas & NumPy
- 🖼 Pillow (PIL)
- 🎤 streamlit-webrtc

## 🧪 Instalasi & Penggunaan

### ⚙️ Prasyarat
- Python 3.8 atau lebih baru
- `pip` (Python package manager)

### 🔧 Langkah Instalasi

1. **Clone repositori:**
   ```bash
   git clone https://github.com/jihanabilah07/UAS_ML_KEL3
   cd rock-paper-scissors-classifier

2. **Buat & aktifkan virtual environment:**

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # MacOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi:**

   ```bash
   streamlit run app.py
   ```

5. **Akses aplikasi di browser:**

   ```
   http://localhost:8501
   ```

## 📂 Struktur Direktori

```
rock-paper-scissors-classifier/
├── app.py                      # Aplikasi utama Streamlit
├── models/
│   └── best_model.h5           # Model CNN terlatih
├── class_mapping.csv           # Pemetaan indeks kelas ke nama kelas
├── requirements.txt            # Daftar dependencies
├── assets/                     # Gambar & aset lainnya
└── README.md                   # Dokumentasi proyek
```

## 🧑‍💻 Cara Menggunakan

### 1️⃣ Upload Image

* Pilih tab **"Upload Image"**
* Klik "Choose an image..." lalu pilih gambar
* Aplikasi akan menampilkan gambar dan hasil klasifikasinya

### 2️⃣ Take Photo

* Pilih tab **"Take Photo"**
* Izinkan akses kamera di browser
* Klik tombol untuk mengambil foto
* Hasil klasifikasi akan ditampilkan

### 3️⃣ Real-time Detection

* Pilih tab **"Real-time Camera Detection"**
* Tunjukkan gestur tangan di depan kamera
* Prediksi muncul secara real-time

## 🧠 Model Machine Learning

Model yang digunakan adalah CNN yang dilatih dengan dataset Rock-Paper-Scissors. Arsitektur model meliputi:

* Beberapa lapisan konvolusi + ReLU
* Max Pooling
* Fully Connected Layer

Model ini mencapai **akurasi \~97%** pada data validasi.

## 🛠 Troubleshooting

| Masalah                | Solusi                                                          |
| ---------------------- | --------------------------------------------------------------- |
| Kamera tidak berfungsi | Pastikan browser mendapat izin & tidak digunakan aplikasi lain  |
| Prediksi tidak akurat  | Pastikan gestur jelas & pencahayaan cukup                       |
| Error saat menjalankan | Pastikan semua dependencies terinstall dengan versi yang sesuai |

## 🤝 Kontribusi

Ingin membantu mengembangkan aplikasi ini?

1. Fork repositori
2. Buat branch baru
3. Ajukan Pull Request
   Kami akan sangat senang menerima kontribusi Anda! 🙌

## 🔗 Link Repositori

[👉 GitHub Repository](https://github.com/jihanabilah07/UAS_ML_KEL3)
