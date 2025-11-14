import streamlit as st
import os
import sys
import pandas as pd

# --- Setup Path ---
# (Penting agar Streamlit bisa menemukan folder 'src')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
if 'src' not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, 'src'))

# --- Impor Modul STKI ---
try:
    import preprocess as pp
    from vsm_ir import VSM
except ImportError:
    st.error("Error: Gagal memuat modul STKI dari folder 'src'. Pastikan file ada.")
    st.stop()

# --- Fungsi Caching Model ---
@st.cache_resource
def load_model_and_data():
    """
    Memuat data dan melatih model VSM.
    Streamlit akan cache hasil fungsi ini agar tidak diulang.
    """
    # 1. Muat data ASLI
    original_corpus = {}
    data_dir = os.path.join(BASE_DIR, 'data')
    for f in os.listdir(data_dir):
        if f.endswith('.txt') and not f.startswith('processed_'):
            with open(os.path.join(data_dir, f), 'r', encoding='utf-8') as file:
                original_corpus[f] = file.read()

    # 2. Muat data PROSES
    processed_corpus = {}
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    for f in os.listdir(processed_dir):
        if f.endswith('.txt'):
            original_doc_id = f.replace('processed_', '')
            with open(os.path.join(processed_dir, f), 'r', encoding='utf-8') as file:
                processed_corpus[original_doc_id] = file.read()
                
    if not processed_corpus:
        st.error("Data yang diproses tidak ditemukan di 'data/processed'.")
        return None, None

    # 3. Latih model VSM (gunakan model sublinear terbaik)
    vsm_model = VSM(sublinear_tf=True)
    vsm_model.fit(processed_corpus, original_corpus)
    
    return vsm_model, original_corpus

# --- Tampilan Aplikasi Streamlit ---
st.set_page_config(page_title="Mesin Pencari STKI", layout="wide")
st.title("🔎 Mesin Pencari STKI (VSM)")
st.subheader("Korpus: 5 Dokumen UIN di Jawa Tengah")

# Muat model
vsm_model, corpus = load_model_and_data()

if vsm_model:
    # --- Sidebar (Input) ---
    with st.sidebar:
        st.header("Kontrol Pencarian")
        
        # Input Kueri
        query = st.text_input(
            "Masukkan Kueri Pencarian:", 
            placeholder="cth: pmb uin walisongo"
        )
        
        # Input Top-k
        k_value = st.slider(
            "Jumlah Hasil (Top-K):", 
            min_value=1, 
            max_value=len(corpus), 
            value=3, 
            step=1
        )
        
        search_button = st.button("Cari Dokumen")
        
        # Tampilkan Dokumen Asli (Opsional)
        st.divider()
        st.write("Dokumen Asli dalam Korpus:")
        for doc_id, text in corpus.items():
            with st.expander(doc_id):
                st.write(text)

    # --- Halaman Utama (Hasil) ---
    if search_button and query:
        st.header(f"Hasil Pencarian Top-{k_value} untuk: '{query}'")
        
        # Lakukan pencarian
        results = vsm_model.search(query, k=k_value)
        
        if not results:
            st.warning("Tidak ada dokumen yang relevan ditemukan.")
        else:
            # Tampilkan hasil (Sesuai Soal 04 - Langkah 4 & Soal 05 - Langkah 3)
            for i, (doc_id, score, snippet) in enumerate(results):
                st.subheader(f"#{i+1}: {doc_id}")
                st.info(f"**Skor (Cosine Similarity): {score:.4f}**")
                
                # Ini adalah "template-based generator" sederhana [cite: 109]
                st.write(f"**Snippet:** {snippet}")
                st.divider()
    
    elif search_button and not query:
        st.error("Silakan masukkan kueri pencarian.")
        
else:
    st.error("Gagal memuat model. Periksa konsol untuk detail.")
