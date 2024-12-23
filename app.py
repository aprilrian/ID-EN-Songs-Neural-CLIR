import streamlit as st
import pandas as pd
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time

@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

@st.cache_data
def load_dataset():
    return pd.read_csv('preprocessed_dataset.csv')

dataset = load_dataset()

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

# Load FAISS Index
try:
    index_multilingual = faiss.read_index("faiss_multilingual.index")
    st.success(f"Indeks FAISS berhasil dimuat dengan {index_multilingual.ntotal} dokumen.")
except Exception as e:
    st.error(f"Error saat memuat indeks FAISS: {e}")
    st.stop()

# Search Function
def search_lyrics(query, k=10):
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
        lyric = dataset.iloc[idx]['preprocessed_lyric'] 
        distance = distances[0][i]
        results.append({"title": title, "lyric": lyric, "distance": distance})
    return results

# Streamlit UI
st.markdown("""
<div style='text-align: center; margin-bottom: 30px;'>
    <h1 style='color: #76C7C0; font-family: Verdana, sans-serif;'>🎵 Pencarian Lirik Lagu 🎵</h1>
    <p style='color: #BBB; font-size: 16px;'>Cari lagu favorit Anda berdasarkan lirik atau kata kunci!</p>
</div>
""", unsafe_allow_html=True)

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
                title = result['title'].title()  # Kapitalisasi judul
                short_lyric = result['lyric'][:300].capitalize() + "..."  # Cuplikan lirik
                full_lyric = result['lyric'].capitalize()  # Lirik lengkap
                distance = result['distance']
        
                # Card Layout
                st.markdown(f"""
                <div style='padding: 15px; margin: 10px 0; border: 1px solid #444; border-radius: 10px; background-color: #2B2B2B; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);'>
                    <h3 style='margin-bottom: 10px; color: #1E90FF; font-family: Arial, sans-serif;'>{i + 1}. {title}</h3>
                    <p style='font-size: 14px; color: #AAA; font-family: Arial, sans-serif;'><b>Distance:</b> {distance:.4f}</p>
                    <p style='font-size: 16px; color: #EEE; font-family: Georgia, serif;'>{short_lyric}</p>
                </div>
                """, unsafe_allow_html=True)
        
                # Gunakan expander untuk lirik lengkap
                with st.expander(f"Tampilkan lirik lengkap untuk '{title}'"):
                    st.write(full_lyric)
            else:
                st.warning("Masukkan kata kunci untuk memulai pencarian!")

st.markdown("<hr style='border: 1px solid #555; margin-top: 30px;'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center;'>
    <p style='color: #777; font-size: 12px;'>Built with ❤️ by TBI MANIACS</p>
</div>
""", unsafe_allow_html=True)
