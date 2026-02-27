import os
import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# Paths (same approach as index builder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))           # .../runtime/rag
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")) # .../mahg-llm-lab
RAG_DIR = os.path.join(REPO_ROOT, "runtime", "rag")

INDEX_PATH = os.path.join(RAG_DIR, "faiss.index")
META_PATH = os.path.join(RAG_DIR, "meta.json")

LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8090/completion")

def load_rag():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        raise RuntimeError("RAG index not found. Run: python runtime/rag/rag_index_build_or_load.py")

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    embed_model = SentenceTransformer(meta["embed_model"])
    chunks = meta["chunks"]  # list of {source, chunk, text}
    return index, embed_model, chunks

def retrieve(index, embed_model, chunks, query, k=5):
    q = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, idx = index.search(q, min(k, len(chunks)))
    hits = [chunks[i] for i in idx[0]]
    return hits

def ask_llama(prompt):
    payload = {
        "prompt": prompt,
        "n_predict": 350,
        "temperature": 0.2,
        "stop": ["<<END>>"]
    }
    r = requests.post(LLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("content", "")

def main():
    question = input("Question: ").strip()
    if not question:
        print("No question provided.")
        return

    index, embed_model, chunks = load_rag()
    hits = retrieve(index, embed_model, chunks, question, k=5)

    print("\n=== RETRIEVED SOURCES ===")
    for h in hits:
        print(f"- {h['source']} (chunk {h['chunk']})")

    context = "\n\n".join([f"[SOURCE: {h['source']} | CHUNK: {h['chunk']}]\n{h['text']}" for h in hits])

    prompt = f"""You are an industrial oxygen production + logistics + finance assistant.
Use ONLY the context below.

Return a concise answer with bullet points.
Add citations at the end of each bullet like: [SOURCE: <path> | CHUNK: <id>]
Finish with: <<END>>

Context:
{context}

Question:
{question}

Answer:
"""

    answer = ask_llama(prompt)
    print("\n===== ANSWER =====\n")
    print(answer.strip())

if __name__ == "__main__":
    main()