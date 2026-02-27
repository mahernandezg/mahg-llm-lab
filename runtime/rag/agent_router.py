import os
import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ---- Paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RAG_DIR = os.path.join(REPO_ROOT, "runtime", "rag")

INDEX_PATH = os.path.join(RAG_DIR, "faiss.index")
META_PATH = os.path.join(RAG_DIR, "meta.json")

LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8090/completion")

# ---- Agent definitions ----
AGENTS = {
    "ops": {
        "title": "Production & Storage Expert",
        "retrieval_hint": "oxygen plant production storage telemetry safety stock dry-out prevention",
        "style": "Focus on production continuity, storage levels, telemetry, redundancy, and operational mitigations."
    },
    "logistics": {
        "title": "Distribution Logistics Expert",
        "retrieval_hint": "emergency routing tanker fleet route optimization ADR compliance priority allocation",
        "style": "Focus on routing, fleet capacity, prioritization, contingency distribution, compliance, and response time."
    },
    "pricing": {
        "title": "Sales & Pricing Expert (Medical LOX)",
        "retrieval_hint": "medical oxygen pricing policy hospital contract fixed price surcharge regulation indexing clauses",
        "style": "Focus on pricing constraints, contract structure, regulatory limits, and commercial implications."
    },
    "controller": {
        "title": "Financial Controller",
        "retrieval_hint": "unit economics cost drivers margin erosion working capital ROCE EBITDA emergency cost modeling",
        "style": "Focus on cost drivers, margin, working capital, KPIs, scenario modeling, and governance/traceability."
    }
}

def choose_agent(question: str) -> str:
    q = question.lower()

    # scoring by keywords (simple + effective for teaching)
    scores = {k: 0 for k in AGENTS.keys()}

    # OPS / production & storage
    for kw in ["plant", "production", "storage", "telemetry", "tank", "dry-out", "redundancy", "safety stock", "shutdown", "asU", "cryogenic"]:
        if kw in q:
            scores["ops"] += 2

    # LOGISTICS
    for kw in ["logistics", "routing", "route", "fleet", "tanker", "delivery", "dispatch", "priority", "allocation", "adr", "gps", "reroute", "transport"]:
        if kw in q:
            scores["logistics"] += 2

    # PRICING
    for kw in ["price", "pricing", "contract", "surcharge", "commercial", "sales", "discount", "indexation", "cpi", "regulatory", "policy", "margin recovery"]:
        if kw in q:
            scores["pricing"] += 2

    # CONTROLLER / finance
    for kw in ["cost", "costs", "ebitda", "roce", "cash", "cashflow", "working capital", "kpi", "p&l", "profit", "margin", "unit economics", "forecast", "budget"]:
        if kw in q:
            scores["controller"] += 2

    # tie-breaker preference: controller > ops > logistics > pricing (you can change)
    order = ["controller", "ops", "logistics", "pricing"]
    best = max(order, key=lambda k: scores[k])

    # if no keywords matched at all, default to ops (or controller)
    if all(v == 0 for v in scores.values()):
        return "ops"

    return best

# ---- RAG load ----
def load_rag():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        raise RuntimeError("RAG index not found. Run: python runtime/rag/rag_index_build_or_load.py")

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    embed_model = SentenceTransformer(meta["embed_model"])
    chunks = meta["chunks"]  # list of {source, chunk, text}
    return index, embed_model, chunks

def retrieve(index, embed_model, chunks, query, k=6):
    q = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, idx = index.search(q, min(k, len(chunks)))
    return [chunks[i] for i in idx[0]]

def ask_llama(prompt):
    payload = {
        "prompt": prompt,
        "n_predict": 260,
        "temperature": 0.2,
        "stop": ["<<END>>"]
    }
    r = requests.post(LLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("content", "")

def build_prompt(agent_key: str, question: str, hits: list):
    agent = AGENTS[agent_key]
    context = "\n\n".join(
        [f"[SOURCE: {h['source']} | CHUNK: {h['chunk']}]\n{h['text']}" for h in hits]
    )

    return f"""You are acting as: {agent['title']}.
{agent['style']}

Use ONLY the context below.

Output requirements:
- Return EXACTLY 7 bullet points.
- No introductions.
- No repeated bullets.
- Do NOT paste context.
- Finish with exactly: <<END>>

Context:
{context}

Question:
{question}

Answer:
"""

def main():
    question = input("Question: ").strip()
    if not question:
        print("No question provided.")
        return

    agent_key = choose_agent(question)
    print(f"\n✅ Selected agent: {agent_key} — {AGENTS[agent_key]['title']}")

    index, embed_model, chunks = load_rag()

    # Two-pass retrieval: agent hint + user question
    q1 = f"{AGENTS[agent_key]['retrieval_hint']} | {question}"
    q2 = question

    hits = []
    seen = set()

    for h in retrieve(index, embed_model, chunks, q1, k=4) + retrieve(index, embed_model, chunks, q2, k=4):
        key = (h["source"], h["chunk"])
        if key not in seen:
            seen.add(key)
            hits.append(h)

    print("\n=== RETRIEVED SOURCES ===")
    for h in hits:
        print(f"- {h['source']} (chunk {h['chunk']})")

    prompt = build_prompt(agent_key, question, hits)
    answer = ask_llama(prompt)

    print("\n===== ANSWER =====\n")
    print(answer.strip())

if __name__ == "__main__":
    main()