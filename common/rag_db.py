# session6/common/rag_simple.py
from __future__ import annotations
import os, math, re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FALLBACK_TXT = os.path.join(DATA_DIR, "support_kb.txt")

def _read_fallback(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _chunk_text(text: str, chunk_size: int = 140) -> list[str]:
    words = text.split()
    chunks, buf = [], []
    for w in words:
        buf.append(w)
        if len(buf) >= chunk_size:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return chunks

def _embed(text: str):
    text = text.lower()
    toks = re.findall(r"\w+", text)
    counts = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    keys = sorted(counts.keys())
    vec = [counts[k] for k in keys]
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v / norm for v in vec], keys

def _cos(a, b):
    L = min(len(a), len(b))
    return sum(a[i]*b[i] for i in range(L))

CORPUS = None

def load_corpus():
    global CORPUS
    txt = _read_fallback(FALLBACK_TXT)
    chunks = _chunk_text(txt)
    docs = []
    for i, ch in enumerate(chunks):
        v, voc = _embed(ch)
        docs.append({"id": f"c{i}", "text": ch, "vec": v, "vocab": voc})
    CORPUS = docs

def rag_search(query: str, k: int = 3):
    global CORPUS
    if CORPUS is None:
        load_corpus()
    qv, qvoc = _embed(query)
    scored = []
    for d in CORPUS:
        scored.append((_cos(qv, d["vec"]), d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, "text": d["text"], "id": d["id"]} for s, d in scored[:k]]

def rag_answer(query: str):
    hits = rag_search(query, k=3)
    if not hits:
        return "No info", 0.0, []
    answer = "\n".join(h["text"] for h in hits)
    return answer, hits[0]["score"], hits
