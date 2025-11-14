import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string

# --- Persiapan NLTK dan Sastrawi ---

# Download data NLTK (hanya perlu sekali)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Siapkan stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Siapkan daftar stopwords Bahasa Indonesia
list_stopwords = stopwords.words('indonesian')
# Tambahkan stopwords kustom jika perlu
custom_stopwords = ['disingkat', 'yg', 'dgn', 'dr', 'thn', 'tsb', 'yakni', 'yaitu']
list_stopwords.extend(custom_stopwords)
list_stopwords = set(list_stopwords)

# --- Fungsi-fungsi Preprocessing Sesuai Soal ---

def clean(text):
    """
    Melakukan case-folding, menghapus angka, tanda baca, dan spasi berlebih.
    Sesuai 'case-folding' dan 'normalisasi angka/tanda baca'.
    """
    # Case folding
    text = text.lower()
    # Hapus angka
    text = re.sub(r"\d+", "", text)
    # Hapus tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Hapus spasi putih berlebih
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def tokenize(text):
    """
    Memecah teks menjadi token (kata).
    Sesuai 'tokenisasi'.
    """
    return word_tokenize(text)

def remove_stopwords(tokens):
    """
    Menghapus stopwords dari daftar token.
    Sesuai 'stopword removal'.
    """
    return [word for word in tokens if word not in list_stopwords]

def stem(tokens):
    """
    Melakukan stemming pada setiap token.
    Sesuai 'stemming/lemmatization'.
    """
    return [stemmer.stem(word) for word in tokens]

def preprocess_text(text):
    """
    Fungsi utama yang menjalankan semua langkah preprocessing.
    """
    cleaned_text = clean(text)
    tokens = tokenize(cleaned_text)
    stopped_tokens = remove_stopwords(tokens)
    stemmed_tokens = stem(stopped_tokens)

    # Mengembalikan sebagai string yang sudah diproses
    return " ".join(stemmed_tokens)

if __name__ == '__main__':
    # Contoh penggunaan modul jika dijalankan langsung
    sample_text = "UIN Walisongo Semarang adalah universitas di Jawa Tengah. Didirikan tahun 1970."

    print("--- Teks Asli ---")
    print(sample_text)

    processed_text = preprocess_text(sample_text)

    print("\n--- Teks Hasil Preprocessing ---")
    print(processed_text)
