import os
import json
import hashlib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Always resolve paths from the repo root (folder that contains this script's parent dirs)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../runtime/rag
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))# .../mahg-llm-lab

KNOWLEDGE_DIR = os.path.join(REPO_ROOT, "knowledge")
RAG_DIR = os.path.join(REPO_ROOT, "runtime", "rag")
INDEX_PATH = os.path.join(RAG_DIR, "faiss.index")
META_PATH = os.path.join(RAG_DIR, "meta.json")

os.makedirs(RAG_DIR, exist_ok=True)

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def scan_knowledge(folder=KNOWLEDGE_DIR):
    files = []
    for root, _, names in os.walk(folder):
        for name in names:
            if name.lower().endswith(".txt"):
                path = os.path.join(root, name)
                size = os.path.getsize(path)
                sha = file_sha256(path)
                files.append({"path": path.replace("\\", "/"), "size": size, "sha256": sha})
    files.sort(key=lambda x: x["path"])
    return files

def chunk_text(text, chunk_size=900, overlap=150):
    chunks = []
    i = 0
    while i < len(text):
        ch = text[i:i+chunk_size].strip()
        if ch:
            chunks.append(ch)
        i += max(1, chunk_size - overlap)
    return chunks

def build_index(files):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks = []  # (source_path, chunk_id, text)
    for f in files:
        path = f["path"]
        with open(path, "r", encoding="utf-8") as fp:
            text = fp.read().strip()
        for j, ch in enumerate(chunk_text(text)):
            chunks.append((path, j, ch))

    if not chunks:
        raise RuntimeError(
            f"No chunks produced. Check that KNOWLEDGE_DIR exists and contains non-empty .txt files.\n"
            f"KNOWLEDGE_DIR = {KNOWLEDGE_DIR}"
        )

    texts = [c[2] for c in chunks]

    emb = embed_model.encode(texts, convert_to_numpy=True).astype(np.float32)
    if emb.ndim == 1:  # single chunk edge-case
        emb = emb.reshape(1, -1)

    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, INDEX_PATH)

    meta = {
        "embed_model": "all-MiniLM-L6-v2",
        "knowledge_dir": KNOWLEDGE_DIR.replace("\\", "/"),
        "files": files,
        "chunks": [{"source": s, "chunk": cid, "text": t} for (s, cid, t) in chunks],
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return index, meta, embed_model

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    embed_model = SentenceTransformer(meta["embed_model"])
    return index, meta, embed_model

def is_cache_valid(current_files, meta):
    return current_files == meta.get("files", [])

def get_index():
    current_files = scan_knowledge()

    print(f"REPO_ROOT     = {REPO_ROOT}")
    print(f"KNOWLEDGE_DIR = {KNOWLEDGE_DIR}")
    print(f"Files found   = {len(current_files)}")

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index, meta, embed_model = load_index()
        if is_cache_valid(current_files, meta):
            print("✅ Using cached FAISS index")
            return index, meta, embed_model
        print("♻️ Knowledge changed, rebuilding index...")

    print("🔨 Building FAISS index...")
    return build_index(current_files)

if __name__ == "__main__":
    get_index()