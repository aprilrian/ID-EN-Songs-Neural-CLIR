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
        artist = dataset.iloc[idx]['artist']
        lyric = dataset.iloc[idx]['preprocessed_lyric']
        lang = dataset.iloc[idx]['lang']
        distance = distances[0][i]
        results.append({"title": title, "artist": artist, "lyric": lyric, "distance": distance, "lang": lang})
    return results

# Streamlit UI
st.markdown("""
<div style='text-align: center; margin-bottom: 30px;'>
    <h1 style='color: #FFFFFF; font-family: "Trebuchet MS", sans-serif; 
    background: linear-gradient(90deg, rgba(118,199,192,1) 0%, rgba(30,144,255,1) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;'>🎵 Indonesia-English Song Lyrics CLIR 🎵</h1>
    <p style='color: #AAA; font-size: 16px;'>Cari lagu favorit Anda berdasarkan cuplikan lirik atau kata kunci!</p>
</div>
""", unsafe_allow_html=True)

# Query Input
query = st.text_input("Masukkan lirik atau kata kunci:")

k = st.slider("Jumlah hasil yang ingin ditampilkan:", min_value=1, max_value=10, value=5)

if st.button("Cari"):
    if query:
        with st.spinner("Sedang mencari..."):
            start_time = time.time()
            results = search_lyrics(query, k=k)
            end_time = time.time()
        
        st.write(f"⏱️ Pencarian selesai dalam {end_time - start_time:.4f} detik.")
        st.write("""
        ### Apa itu Proximitas?
        **Proximitas** adalah ukuran jarak antara kata kunci pencarian Anda dan hasil yang ditemukan. 
        Semakin kecil nilainya, semakin sesuai hasil pencarian dengan yang Anda cari.
        """)
        
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
                <div style='padding: 20px; margin: 10px 0; border: 1px solid #76C7C0; 
                border-radius: 10px; background: linear-gradient(135deg, #2C2C2C, #1E1E1E); 
                box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.6);'>
                    <h3 style='margin-bottom: 10px; color: #76C7C0; font-family: "Arial Black", sans-serif;'>{i + 1}. {title}</h3>
                    <div style='margin-bottom: 10px;'>
                        <p style='font-size: 14px; color: #FFD700; font-family: Arial, sans-serif; margin: 5px 0;'><b>Proximitas:</b> {result['distance']:.4f} <span style='font-size: 12px; color: #76C7C0;'>(Semakin kecil, semakin relevan)</span></p>
                        <p style='font-size: 14px; color: #AAA; font-family: Arial, sans-serif; margin: 2px 0;'><b>Artis:</b> {result['artist']}</p>
                        <p style='font-size: 14px; color: #AAA; font-family: Arial, sans-serif; margin: 2px 0;'><b>Bahasa:</b> {"Indonesia" if result['lang'] == "id" else "Inggris"}</p>
                    </div>
                    <p style='font-size: 16px; color: #DDD; font-family: Georgia, serif; margin-top: 10px;'>{short_lyric}</p>
                </div>
                """, unsafe_allow_html=True)

                # Gunakan expander untuk lirik lengkap
                with st.expander(f"🎵 Klik untuk melihat lirik lengkap dari '{title}' oleh {result['artist']}"):
                    st.markdown(f"""
                    <div style='max-height: 300px; overflow-y: auto; padding: 15px; 
                    background: #2C2C2C; border-radius: 8px; border: 1px solid #76C7C0;'>
                        <p style='color: #EEE; font-size: 14px; line-height: 1.6;'>{full_lyric}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("Masukkan kata kunci untuk memulai pencarian!")

st.markdown("<hr style='border: 1px solid #555; margin-top: 30px;'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-top: 50px;'>
    <p style='color: #555; font-size: 14px;'>Built with ❤️ by <b>TBI MANIACS</b>.</p>
    <a href="https://github.com/aprilrian/ID-EN-Songs-Neural-CLIR" style='color: #76C7C0; text-decoration: none;'>GitHub Repo</a>
</div>
""", unsafe_allow_html=True)
