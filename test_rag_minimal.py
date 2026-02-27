import os
import numpy as np
import requests
import faiss
from sentence_transformers import SentenceTransformer

KNOWLEDGE_DIR = "knowledge"
LLAMA_URL = "http://localhost:8090/completion"

def load_text_files(folder=KNOWLEDGE_DIR):
    files = []
    for root, _, names in os.walk(folder):
        for name in names:
            if name.lower().endswith(".txt"):
                path = os.path.join(root, name)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        files.append((path, text))
    return files

def chunk_text(text, chunk_size=900, overlap=150):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks

# 1) Load + chunk
raw_files = load_text_files()
print("\n=== INDEXED FILES ===")
for p, t in raw_files:
    print("-", p, f"(chars={len(t)})")

if not raw_files:
    raise RuntimeError(f"No non-empty .txt files under '{KNOWLEDGE_DIR}/'.")

chunks = []  # (source_path, chunk_id, chunk_text)
for path, text in raw_files:
    for j, ch in enumerate(chunk_text(text)):
        chunks.append((path, j, ch))

texts = [c[2] for c in chunks]

# 2) Embed + index
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
emb = embed_model.encode(texts, convert_to_numpy=True)
if emb.ndim == 1:
    emb = emb.reshape(1, -1)

index = faiss.IndexFlatL2(emb.shape[1])
index.add(emb.astype(np.float32))

# 3) Retrieve top-K chunks
def retrieve(query, k=4):
    q = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, idx = index.search(q, min(k, len(texts)))
    hits = []
    for i in idx[0]:
        src, cid, txt = chunks[i]
        hits.append((src, cid, txt))
    return hits

# 4) Call llama.cpp
def ask_llama(prompt):
    payload = {
    "prompt": prompt,
    "n_predict": 500,
    "temperature": 0.2,
    "stop": ["<<END>>"]
    }
    r = requests.post(LLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json().get("content", "")

question = (
  "Madrid plant shutdown + demand surge: provide (1) routing mitigation, "
  "(2) top emergency cost drivers, and (3) unit-economics + pricing policy implications."
)

hits_ops = retrieve("LOX hospital emergency routing mitigation during plant shutdown", k=3)
hits_fin = retrieve("financial impact emergency oxygen deliveries cost drivers margin working capital", k=3)

# merge + de-dup
seen = set()
hits = []
for src, cid, txt in hits_ops + hits_fin:
    key = (src, cid)
    if key not in seen:
        seen.add(key)
        hits.append((src, cid, txt))

print("\n=== RETRIEVED CHUNKS ===")
for src, cid, _ in hits:
    print(f"- {src} (chunk {cid})")

context = "\n\n".join([f"[SOURCE: {src} | CHUNK: {cid}]\n{txt}" for src, cid, txt in hits])

def is_valid_bullets(text: str) -> bool:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) != 8:
        return False
    for i, ln in enumerate(lines, start=1):
        if not ln.startswith(f"B{i}:"):
            return False
        if not re.search(r"\[SOURCE: .+ \| CHUNK: \d+\]$", ln):
            return False
    return True

def repair_format(bad_text: str, context: str, question: str) -> str:
    fix_prompt = f"""Fix the format ONLY.

You MUST output exactly 8 lines:
B1..B8, each ending with: [SOURCE: <path> | CHUNK: <id>]
Do NOT add extra text. Do NOT repeat the question. Do NOT paste context.
Then print exactly: <<END>>

Context:
{context}

Question:
{question}

Bad Answer (to fix):
{bad_text}

Fixed Answer:
"""
    return ask_llama(fix_prompt)

final_prompt = f"""You are an industrial oxygen production + logistics + finance assistant.
Use ONLY the context below.

Return exactly 8 lines, formatted EXACTLY like:
B1: <text> [SOURCE: <path> | CHUNK: <id>]
...
B8: <text> [SOURCE: <path> | CHUNK: <id>]
Then print exactly: <<END>>

Rules:
- Exactly 8 lines (B1..B8).
- Each line MUST end with one citation tag.
- Do NOT repeat the question.
- Do NOT paste context.

Context:
{context}

Question:
{question}

Answer:
"""

answer = ask_llama(final_prompt).strip()

if not is_valid_bullets(answer):
    print("\n===== REPAIRING FORMAT =====\n")
    answer = repair_format(answer, context, question).strip()

print("\n===== ANSWER =====\n")
print(answer)