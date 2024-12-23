import streamlit as st
import pandas as pd
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time

# Load Dataset
try:
    dataset = pd.read_csv('preprocessed_dataset.csv')
    if dataset.empty:
        st.error("Dataset kosong! Pastikan dataset tersedia.")
        st.stop()
except Exception as e:
    st.error(f"Error saat memuat dataset: {e}")
    st.stop()

# Preprocessing Function
def preprocess(lyric):
    lyric = lyric.lower()  # Lowercase
    lyric = re.sub(r'\[.*?\]|[^a-zA-Z0-9\s]', ' ', lyric).strip()  # Remove special characters
    return lyric

def preprocess_query(query):
    return preprocess(query)

# Normalization Function
def normalize(embedding):
    norm = np.linalg.norm(embedding)
    return embedding if norm == 0 else embedding / norm

# Load Model
try:
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

# Load FAISS Index
try:
    index_multilingual = faiss.read_index("faiss_multilingual.index")
    st.success(f"Indeks FAISS berhasil dimuat dengan {index_multilingual.ntotal} dokumen.")
except Exception as e:
    st.error(f"Error saat memuat indeks FAISS: {e}")
    st.stop()

# Search Function
def search_lyrics(query, k=5):
    query = preprocess_query(query)
    if not query.strip():
        return "Kueri kosong. Masukkan kata kunci."
    
    try:
        query_embedding = model.encode([query], convert_to_tensor=False)
        query_embedding = normalize(query_embedding)
        distances, indices = index_multilingual.search(
            np.array(query_embedding, dtype='float32').reshape(1, -1), k
        )
    except Exception as e:
        return f"Error saat pencarian: {e}"
    
    if len(indices[0]) == 0:
        return "Tidak ada hasil yang ditemukan."

    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        title = dataset.iloc[idx]['title']
        lyric = dataset.iloc[idx]['lyric'][:300]  # Limit lyric preview
        distance = distances[0][i]
        results.append({"title": title, "lyric": lyric, "distance": distance})
    return results

# Streamlit UI
st.title("Pencarian Lirik Lagu 🎵")
st.write("Cari lagu favorit Anda berdasarkan lirik atau kata kunci!")

# Query Input
query = st.text_input("Masukkan lirik atau kata kunci:")

if st.button("Cari"):
    if query:
        with st.spinner("Sedang mencari..."):
            start_time = time.time()
            results = search_lyrics(query)
            end_time = time.time()
        
        st.write(f"⏱️ Pencarian selesai dalam {end_time - start_time:.4f} detik.")
        
        if isinstance(results, str):
            st.error(results)
        else:
            for i, result in enumerate(results):
                with st.container():
                    st.write(f"### {i + 1}. {result['title']}")
                    st.write(f"**Distance:** {result['distance']:.4f}")
                    st.write(f"*{result['lyric']}...*")
                    st.markdown("---")
    else:
        st.warning("Masukkan kata kunci untuk memulai pencarian!")
