# utils.py
import requests
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import time

VLLM_API_URL = os.environ.get("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
CHROMA_DIR = "chroma_db"
DOC_COLLECTION = "enterprise_docs"
LT_MEMORY_COLLECTION = "longterm_memory"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 初始化 chroma client & embedding model
_chroma_client = None
_embed_model = None

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
    return _chroma_client

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

def retrieve_docs(query, k=4):
    """从 enterprise_docs 中检索 top-k 相关片段（返回文本 + metadata）"""
    client = get_chroma_client()
    try:
        coll = client.get_collection(DOC_COLLECTION)
    except Exception:
        return []
    embed_model = get_embed_model()
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    res = coll.query(query_embeddings=[q_emb], n_results=k)
    docs = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        docs.append({"text": doc, "meta": meta})
    return docs

def save_to_longterm(text, meta=None):
    """将重要信息存入长期记忆 collection"""
    client = get_chroma_client()
    try:
        coll = client.get_collection(LT_MEMORY_COLLECTION)
    except Exception:
        coll = client.create_collection(name=LT_MEMORY_COLLECTION, metadata={})
    embed_model = get_embed_model()
    emb = embed_model.encode([text], convert_to_numpy=True)[0]
    coll.add(ids=[f"lt_{int(time.time()*1000)}"], documents=[text], metadatas=[meta or {}], embeddings=[emb])
    client.persist()

def retrieve_longterm(query, k=3):
    client = get_chroma_client()
    try:
        coll = client.get_collection(LT_MEMORY_COLLECTION)
    except Exception:
        return []
    embed_model = get_embed_model()
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    res = coll.query(query_embeddings=[q_emb], n_results=k)
    docs = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        docs.append({"text": doc, "meta": meta})
    return docs

def call_vllm_chat(system_prompt, messages, temperature=0.2, max_tokens=512):
    """
    调用 vLLM 的 OpenAI-compatible chat completions API。
    messages: List[{"role":"system|user|assistant", "content": "..."}]
    返回文本（str）
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",   # server 端需要支持该 model id 或忽略
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    # vLLM openai-compatible endpoint usually at /v1/chat/completions
    resp = requests.post(VLLM_API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    # 兼容不同返回格式：尝试提取第一个 choice 的 message.content
    try:
        text = j["choices"][0]["message"]["content"]
    except Exception:
        # 退回到可能的 fields
        text = str(j)
    return text
