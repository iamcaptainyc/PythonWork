# ingest.py
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000   # 每个 chunk 最大字符数（可调）
CHUNK_OVERLAP = 200

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def load_texts_from_dir(data_dir=DATA_DIR):
    docs = []
    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(".txt"):
            continue
        path = os.path.join(data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            docs.append({
                "id": f"{filename}__{i}",
                "text": c,
                "meta": {"source": filename, "chunk_index": i}
            })
    return docs

def main():
    # 初始化 Chroma 客户端（本地持久化）
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
    collection_name = "enterprise_docs"
    # If exists, drop or get existing
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name, metadata={})

    # embedding function wrapper using sentence-transformers
    model = SentenceTransformer(EMBED_MODEL_NAME)
    def embed_fn(texts):
        return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # load and chunk
    docs = load_texts_from_dir(DATA_DIR)
    if not docs:
        print("No .txt files under data/ to ingest.")
        return

    ids = [d["id"] for d in docs]
    metadatas = [d["meta"] for d in docs]
    texts = [d["text"] for d in docs]

    # Add to chroma
    collection.add(ids=ids, metadatas=metadatas, documents=texts, embeddings=embed_fn(texts))
    client.persist()
    print(f"Ingested {len(texts)} chunks into Chroma collection '{collection_name}'.")

if __name__ == "__main__":
    main()
