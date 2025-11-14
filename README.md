# Proyek UTS STKI (A11.4703) - Mini Search Engine

[cite_start]Ini adalah proyek Ujian Tengah Semester (UTS) Ganjil 2025/2026 untuk mata kuliah **Sistem Temu Kembali Informasi (STKI)** [cite: 12] di Universitas Dian Nuswantoro.

**Author:**
* **Nama:** [SILVAN RIDHO PRADANA]
* **NIM:** [A11.2022.14284]
* **Kelompok:** A11.4703 
* **Dosen:** Abu Salam, M.Kom
* **Streamlit:** https://uts-stki-14284.streamlit.app/


---

## 1. Deskripsi Proyek

Proyek ini adalah implementasi *mini search engine* yang mencakup alur kerja STKI fundamental. Tujuannya adalah untuk membangun sistem dari awal yang dapat:
1.  Melakukan *preprocessing* pada korpus dokumen teks.
2.  Membangun model *Boolean Retrieval* dengan Inverted Index.
3.  Membangun model *Vector Space Model (VSM)* dengan pembobotan TF-IDF.
4.  Memberikan *ranking* hasil pencarian berdasarkan Cosine Similarity.
5.  Mengevaluasi performa model menggunakan metrik seperti Precision@k dan MAP@k.

Korpus data yang digunakan adalah 5 dokumen `.txt` buatan sendiri yang berisi informasi tentang Universitas Islam Negeri (UIN) di Jawa Tengah.

## 2. Fitur

* [cite_start]**Document Preprocessing (Soal 02):** Implementasi modul `preprocess.py` yang mencakup *case-folding*, *tokenisasi*, *stopword removal* (NLTK), dan *stemming* (Sastrawi)[cite: 57].
* [cite_start]**Boolean Retrieval Model (Soal 03):** Implementasi modul `boolean_ir.py` yang membangun **Inverted Index** dan dapat memproses kueri `AND`, `OR`, dan `NOT`[cite: 66, 73].
* [cite_start]**Vector Space Model (VSM) (Soal 04):** Implementasi modul `vsm_ir.py` menggunakan `scikit-learn` untuk membangun matriks **TF-IDF** dan menghitung **Cosine Similarity** untuk perankingan top-k[cite: 91, 94, 96].
* [cite_start]**Evaluasi & Perbandingan (Soal 05):** Membandingkan dua skema pembobotan (TF-IDF standar vs TF-IDF Sublinear) [cite: 102] [cite_start]dan mengevaluasinya menggunakan `Precision@k` dan `MAP@k`[cite: 112].
* [cite_start]**Antarmuka Web (Permintaan Pengguna & Soal 05):** Aplikasi **Streamlit** interaktif (`app/main.py`) untuk memudahkan pencarian menggunakan model VSM[cite: 107].
* [cite_start]**Orchestrator CLI (Soal 05):** Skrip `src/search.py` yang dapat dijalankan via *command-line* untuk memanggil model Boolean atau VSM [cite: 104-106].

## 3. Struktur Folder
```
stki-uts-[nim]-[nama]/
├── app/
│ └── main.py # Skrip utama aplikasi Streamlit
├── data/
│ ├── processed/ # Folder untuk .txt hasil preprocessing (output Soal 02)
│ ├── doc1.txt # Dokumen korpus 1 (UIN Walisongo)
│ ├── doc2.txt # Dokumen korpus 2 (UIN RMS)
│ ├── doc3.txt # Dokumen korpus 3 (UIN Gus Dur)
│ ├── doc4.txt # Dokumen korpus 4 (UIN Salatiga)
│ └── doc5.txt # Dokumen korpus 5 (Info PMB)
├── notebooks/
│ └── UTS_STKI_[nim].ipynb # Notebook utama (Google Colab) untuk pengerjaan & evaluasi
├── reports/
│ └── laporan.pdf # Laporan akhir (jawaban Soal 1-5)
├── src/
│ ├── preprocess.py # Modul preprocessing (Soal 02)
│ ├── boolean_ir.py # Modul Boolean IR (Soal 03)
│ ├── vsm_ir.py # Modul VSM IR (Soal 04)
│ └── search.py # CLI orchestrator (Soal 05)
├── readme.md # File ini
└── requirements.txt # Daftar pustaka Python

```


## 4. Instalasi

Proyek ini dikembangkan menggunakan **Python** (disarankan versi 3.10+).

1.  **Clone Repositori:**
    ```bash
    git clone [URL_REPO_ANDA]
    cd stki-uts-[nim]-[nama]
    ```

2.  **(Opsional) Buat Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    .\venv\Scripts\activate   # Windows
    ```

3.  **Install Pustaka (Requirements):**
    [cite_start]Semua pustaka yang dibutuhkan ada di `requirements.txt`[cite: 142].
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Data NLTK:**
    Modul `preprocess.py` membutuhkan data `punkt` (untuk tokenisasi) dan `stopwords` (untuk stopword removal). Jalankan Python shell dan ketik:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

## 5. Cara Menjalankan

### 5.1. Menjalankan Notebook Eksperimen (Google Colab / Jupyter)

Cara utama untuk melihat proses pengerjaan adalah melalui *notebook*.

1.  Buka dan jalankan file `notebooks/UTS_STKI_[nim].ipynb`.
2.  **PENTING:** Jalankan sel-sel di notebook secara berurutan. Sel **Soal 02** **wajib** dijalankan terlebih dahulu untuk membuat file di `data/processed/`, yang akan digunakan oleh Soal 03, 04, dan 05.
3.  Notebook ini berisi semua langkah, mulai dari preprocessing, uji model Boolean, uji model VSM, hingga evaluasi perbandingan skema bobot.

### 5.2. Menjalankan Antarmuka Web (Streamlit)

Ini adalah antarmuka pengguna utama untuk mendemokan model VSM (pembobotan sublinear).

1.  Pastikan Anda berada di *root folder* proyek (`stki-uts-[nim]-[nama]/`).
2.  Jalankan perintah berikut di terminal Anda:
    ```bash
    streamlit run app/main.py
    ```
3.  Buka browser Anda dan akses `http://localhost:8501`.

### 5.3. Menjalankan CLI Orchestrator (Soal 05)

[cite_start]Ini adalah skrip `search.py` [cite: 104] yang memenuhi **Soal 05 (Langkah 2)**.

1.  Pastikan Anda berada di *root folder* proyek.
2.  Gunakan `python src/search.py --help` untuk melihat opsi.

**Contoh Penggunaan VSM:**
```bash
# Mencari top 2 hasil untuk kueri "info pmb"
python src/search.py --model vsm --k 2 --query "info pmb"

