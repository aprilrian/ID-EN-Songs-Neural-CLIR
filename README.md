# ID-EN-Songs-Neural-CLIR

Multilingual lyric search system (Indonesian ↔ English) using neural Cross-Lingual Information Retrieval (CLIR).  
Embeddings: MiniLM-L12-v2. Index: FAISS. Ranking: Euclidean distance.

## Results
- Precision@3: **91%**
- F1@3: **88%**

## Features
- Cross-language lyric retrieval (ID ↔ EN)
- Fast approximate nearest neighbor search with FAISS
- End-to-end pipeline: preprocessing → embedding → indexing → evaluation
- Simple REST demo using FastAPI

## Installation
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Preprocess data
```bash
python src/preprocess.py --in data/corpus.jsonl --out data/corpus.clean.jsonl
```

### Generate embeddings
```bash
python src/embed.py   --in data/corpus.clean.jsonl   --text_key lyrics   --model sentence-transformers/all-MiniLM-L12-v2   --out data/corpus.embeddings.npy
```

### Build FAISS index
```bash
python src/index_faiss.py build   --embeddings data/corpus.embeddings.npy   --index_out data/index.faiss   --meta_out data/meta.pkl   --index_type flat
```

### Evaluate
```bash
python src/evaluate.py   --queries data/queries.jsonl   --qrels data/qrels.jsonl   --index data/index.faiss   --meta data/meta.pkl   --model sentence-transformers/all-MiniLM-L12-v2   --topk 10
```

Expected metrics: **Precision@3 ≈ 0.91**, **F1@3 ≈ 0.88**.

### Run API demo
```bash
uvicorn src.api_demo:app --host 0.0.0.0 --port 8000
```
Sample request:
```json
POST /search
{ "query": "lagu tentang hujan", "lang": "id", "topk": 5 }
```

## Limitations
- Dataset not included due to copyright (provide your own lyric corpus).
- Performance may degrade with slang or code-mixed queries.
- Genre/domain shifts may require re-tuning.

## Roadmap
- Hybrid BM25 + embeddings
- Reranker with cross-encoder
- Better handling of slang/code-mixing
- Dockerfile and CI/CD integration

## Citation
```
@software{IDEN_Songs_Neural_CLIR,
  title   = {ID-EN-Songs-Neural-CLIR},
  author  = {Rian Aprilyanto Siburian},
  year    = {2024},
  url     = {https://github.com/<your-username>/ID-EN-Songs-Neural-CLIR}
}
```

## License
MIT License. See `LICENSE` for details.

## Acknowledgements
- [sentence-transformers](https://www.sbert.net/)  
- [FAISS](https://github.com/facebookresearch/faiss)  
- [FastAPI](https://fastapi.tiangolo.com/)
