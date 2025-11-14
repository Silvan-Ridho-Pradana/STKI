import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Tambahkan path ke 'src' untuk impor 'preprocess'
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import preprocess as pp

class VSM:
    def __init__(self, sublinear_tf=False):
        """
        Inisialisasi model VSM.
        Args:
            sublinear_tf (bool): Apakah akan menggunakan pembobotan TF sublinear (log(tf) + 1).
                                 Ini untuk memenuhi Soal 05 - Langkah 1.
        """
        self.vectorizer = TfidfVectorizer(
            use_idf=True, 
            smooth_idf=True, 
            norm='l2',
            sublinear_tf=sublinear_tf # Parameter perbandingan
        )
        self.doc_vectors = None
        self.doc_ids = []
        self.original_docs = {}
        self.model_name = "Sublinear TF-IDF" if sublinear_tf else "Standard TF-IDF"

    def fit(self, processed_corpus, original_corpus):
        """
        Melatih (fit) vectorizer pada korpus.
        """
        self.doc_ids = list(processed_corpus.keys())
        corpus_texts = [processed_corpus[doc_id] for doc_id in self.doc_ids]
        self.original_docs = original_corpus
        self.doc_vectors = self.vectorizer.fit_transform(corpus_texts)
        
        # print(f"VSM model ({self.model_name}) fitted. Shape: {self.doc_vectors.shape}")

    def search(self, query, k=3):
        """
        Mencari dokumen yang paling relevan dengan kueri.
        """
        if self.doc_vectors is None:
            raise Exception("Model VSM belum di-fit. Jalankan .fit() terlebih dahulu.")
            
        processed_query = pp.preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])
        cosine_scores = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        k = min(k, len(self.doc_ids)) 
        top_indices = np.argsort(cosine_scores)[::-1]
        
        results = []
        for i in range(k):
            doc_index = top_indices[i]
            doc_id = self.doc_ids[doc_index]
            score = cosine_scores[doc_index]
            
            if score < 1e-9:
                continue
                
            original_text = self.original_docs[doc_id]
            snippet = original_text.replace('\n', ' ')[:120] + "..."
            results.append((doc_id, score, snippet))
            
        return results
