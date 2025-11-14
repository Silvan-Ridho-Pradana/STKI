import os
import sys
import argparse
import pandas as pd

# --- Setup Path ---
# (Penting agar bisa import modul dari 'src' saat dijalankan dari root)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)
    
# Tambahkan root directory (satu level di atas 'src') ke path
root_dir = os.path.abspath(os.path.join(src_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# --- Impor Modul STKI ---
try:
    import preprocess as pp
    import boolean_ir as bir
    from vsm_ir import VSM
except ImportError as e:
    print(f"Error: Gagal mengimpor modul STKI. Pastikan file ada di folder 'src'.")
    print(f"Detail: {e}")
    sys.exit(1)

def load_data(data_dir='data', processed_dir='data/processed'):
    """Memuat data asli dan data yang sudah diproses."""
    
    # 1. Muat data ASLI
    original_corpus = {}
    try:
        for f in os.listdir(data_dir):
            if f.endswith('.txt') and not f.startswith('processed_'):
                with open(os.path.join(data_dir, f), 'r', encoding='utf-8') as file:
                    original_corpus[f] = file.read()
    except FileNotFoundError:
        print(f"Error: Folder data asli '{data_dir}' tidak ditemukan.")
        sys.exit(1)

    # 2. Muat data PROSES
    processed_corpus = {}
    try:
        for f in os.listdir(processed_dir):
            if f.endswith('.txt'):
                original_doc_id = f.replace('processed_', '')
                with open(os.path.join(processed_dir, f), 'r', encoding='utf-8') as file:
                    processed_corpus[original_doc_id] = file.read()
    except FileNotFoundError:
        print(f"Error: Folder data proses '{processed_dir}' tidak ditemukan.")
        print("Jalankan notebook Soal 02 terlebih dahulu.")
        sys.exit(1)
        
    return original_corpus, processed_corpus

def main():
    """Fungsi utama untuk orkestrasi search engine CLI."""
    
    # --- Parser Argumen CLI (Sesuai Soal 05 - Langkah 2) ---
    parser = argparse.ArgumentParser(description="STKI Search Engine Orchestrator")
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['boolean', 'vsm'], 
        required=True,
        help="Model retrieval yang akan digunakan (boolean atau vsm)."
    )
    parser.add_argument(
        '--query', 
        type=str, 
        required=True,
        help="Teks kueri yang akan dicari."
    )
    parser.add_argument(
        '--k', 
        type=int, 
        default=3,
        help="Jumlah top-k dokumen yang ditampilkan (hanya untuk VSM)."
    )
    args = parser.parse_args()

    # --- Muat Data dan Latih Model ---
    # print("Memuat data...")
    original_corpus, processed_corpus = load_data()
    
    if not processed_corpus:
        print("Tidak ada data yang ditemukan. Eksekusi dibatalkan.")
        return

    # print("Mempersiapkan model...")
    all_doc_ids = set(processed_corpus.keys())
    
    # Siapkan model Boolean
    inverted_index = bir.build_inverted_index(processed_corpus)
    
    # Siapkan model VSM (gunakan model sublinear terbaik)
    vsm_model = VSM(sublinear_tf=True)
    vsm_model.fit(processed_corpus, original_corpus)

    print(f"\n--- Hasil Pencarian ---")
    print(f"Model : {args.model}")
    print(f"Query : '{args.query}'")

    # --- Eksekusi Model yang Dipilih ---
    if args.model == 'boolean':
        results, explanation = bir.process_query(args.query, inverted_index, all_doc_ids)
        print(f"Penjelasan : {explanation}")
        if results:
            print(f"Dokumen ditemukan ({len(results)}):")
            for doc_id in sorted(list(results)):
                print(f"  - {doc_id}")
        else:
            print("Tidak ada dokumen yang ditemukan.")
            
    elif args.model == 'vsm':
        results = vsm_model.search(args.query, k=args.k)
        print(f"K (Top-k) : {args.k}")
        if results:
            print(f"Dokumen ditemukan ({len(results)}):")
            output_data = []
            for doc_id, score, snippet in results:
                output_data.append({
                    'Doc_ID': doc_id,
                    'Score (Cosine)': f"{score:.4f}",
                    'Snippet': snippet
                })
            df = pd.DataFrame(output_data)
            print(df.to_string(index=False))
        else:
            print("Tidak ada dokumen yang relevan ditemukan.")

if __name__ == "__main__":
    main()
