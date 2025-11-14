import os
import sys

# Tambahkan path ke 'src' untuk impor 'preprocess'
# (Berguna jika modul ini dipanggil dari notebook di root)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import preprocess as pp

def build_inverted_index(processed_corpus):
    """
    Membangun inverted index dari korpus yang sudah diproses.
    Format: {term: {doc_id, doc_id_2, ...}, ...}
    Sesuai Soal 03 - Langkah 1 & 2
    """
    inverted_index = {}
    for doc_id, text in processed_corpus.items():
        tokens = text.split()
        for token in set(tokens): # Gunakan set() untuk menghitung tiap kata hanya sekali per dokumen
            if token not in inverted_index:
                inverted_index[token] = set()
            inverted_index[token].add(doc_id)
    return inverted_index

def process_query(query, index, all_doc_ids):
    """
    Memproses kueri Boolean sederhana (AND, OR, NOT).
    Sesuai Soal 03 - Langkah 3
    
    Contoh: 
    - "semarang"
    - "semarang AND walisongo"
    - "salatiga OR pekalongan"
    - "walisongo AND NOT salatiga"
    """
    
    # Preprocess kueri sama seperti dokumen
    processed_query = pp.preprocess_text(query)
    tokens = processed_query.split()
    
    # Jika hanya satu kata
    if len(tokens) == 1 and tokens[0] not in ["AND", "OR", "NOT"]:
        term = tokens[0]
        results = index.get(term, set())
        explanation = f"Hasil untuk '{term}': {len(results)} dokumen ditemukan."
        return results, explanation

    # Logika untuk kueri kompleks (disederhanakan untuk 3 token)
    # Format: term1 OPERATOR term2  ATAU  OPERATOR term1
    
    results = set()
    explanation = "Query tidak valid."

    try:
        if "AND NOT" in processed_query:
            parts = processed_query.split(" AND NOT ")
            term1 = parts[0].strip()
            term2 = parts[1].strip()
            
            set1 = index.get(term1, set())
            set2 = index.get(term2, set())
            results = set1.difference(set2) # Interseksi dengan komplemen (A AND (NOT B))
            
            explanation = f"Mencari '{term1}' (ditemukan {len(set1)}) DAN BUKAN '{term2}' (ditemukan {len(set2)}). Hasil (irisan komplemen): {len(results)} dokumen."

        elif "AND" in processed_query:
            parts = processed_query.split(" AND ")
            term1 = parts[0].strip()
            term2 = parts[1].strip()

            set1 = index.get(term1, set())
            set2 = index.get(term2, set())
            results = set1.intersection(set2) # Irisan (Intersection)
            
            explanation = f"Mencari '{term1}' (ditemukan {len(set1)}) DAN '{term2}' (ditemukan {len(set2)}). Hasil (irisan): {len(results)} dokumen."

        elif "OR" in processed_query:
            parts = processed_query.split(" OR ")
            term1 = parts[0].strip()
            term2 = parts[1].strip()

            set1 = index.get(term1, set())
            set2 = index.get(term2, set())
            results = set1.union(set2) # Gabungan (Union)
            
            explanation = f"Mencari '{term1}' (ditemukan {len(set1)}) ATAU '{term2}' (ditemukan {len(set2)}). Hasil (gabungan): {len(results)} dokumen."
        
        elif "NOT" in processed_query:
            term = processed_query.split("NOT ")[1].strip()
            set1 = index.get(term, set())
            results = all_doc_ids.difference(set1) # Komplemen
            
            explanation = f"Mencari SEMUA DOKUMEN (total {len(all_doc_ids)}) KECUALI yang mengandung '{term}' (ditemukan {len(set1)}). Hasil (komplemen): {len(results)} dokumen."

    except Exception as e:
        return set(), f"Error memproses query: {e}. Gunakan format: 'term1 AND term2', 'term1 OR term2', 'term1 AND NOT term2', atau 'NOT term1'."

    return results, explanation

if __name__ == '__main__':
    # Contoh penggunaan jika dijalankan langsung
    dummy_corpus = {
        'docA': 'ini adalah dokumen pertama',
        'docB': 'ini adalah dokumen kedua',
        'docC': 'dokumen pertama dan kedua'
    }
    dummy_all_docs = set(dummy_corpus.keys())
    
    index = build_inverted_index(dummy_corpus)
    print("--- Inverted Index ---")
    print(index)
    
    print("\n--- Tes Query ---")
    query = "dokumen AND pertama"
    results, explanation = process_query(query, index, dummy_all_docs)
    print(f"Query: '{query}' -> Hasil: {results}\nPenjelasan: {explanation}")
