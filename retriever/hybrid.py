# retriever/hybrid.py
import os, json, pickle
import numpy as np
import faiss
from collections import defaultdict

def load_index(index_dir):
    with open(os.path.join(index_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    embed_backend = manifest.get("embed_backend","tfidf")
    vectorizer = None
    index = None
    if embed_backend == "sbert":
        # dense.faiss + optional tiny tfidf.pkl (fallback features)
        index = faiss.read_index(os.path.join(index_dir, "dense.faiss"))
        # we also keep tfidf.pkl around as fallback features; optional to load here
        try:
            with open(os.path.join(index_dir, "tfidf.pkl"), "rb") as f:
                vectorizer = pickle.load(f)
        except Exception:
            vectorizer = None
    else:
        vectorizer = pickle.load(open(os.path.join(index_dir, "tfidf.pkl"), "rb"))
        index = faiss.read_index(os.path.join(index_dir, "tfidf.faiss"))

    chunks = manifest.get("chunks", [])
    pages = manifest.get("pages", [])
    return chunks, pages, bm25, vectorizer, index

def hybrid_search(query, chunks, bm25, vectorizer, faiss_index, embed_backend="tfidf", encoder=None, top_k=50):
    if len(chunks) == 0:
        return []
    # BM25 (lexical)
    bm25_scores = bm25.get_scores(query.lower().split())

    # Dense / vector search
    if embed_backend == "sbert":
        if encoder is None:
            # no dense signal, fall back to lexical only
            dense_scores = np.zeros(len(chunks), dtype=np.float32)
        else:
            qv = encoder([query])  # expected shape (1, d), L2-normalized
            D, I = faiss_index.search(qv.astype(np.float32), min(top_k*4, len(chunks)))
            dense_scores = np.zeros(len(chunks), dtype=np.float32)
            if I.shape[1] > 0:
                dense_scores[I[0]] = D[0]
    else:
        # TF-IDF cosine via vectorizer
        qv = vectorizer.transform([query]).astype(np.float32).toarray()
        qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
        D, I = faiss_index.search(qv, min(top_k*4, len(chunks)))
        dense_scores = np.zeros(len(chunks), dtype=np.float32)
        if I.shape[1] > 0:
            dense_scores[I[0]] = D[0]

    def norm(x):
        x = np.array(x, dtype=np.float32)
        mx, mn = x.max(), x.min()
        if mx == mn:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)

    h = 0.60*norm(dense_scores) + 0.40*norm(bm25_scores)
    idxs = np.argsort(-h)[:top_k]
    return [{"idx": int(i), "score": float(h[i]), "page": chunks[i]["page"], "text": chunks[i]["text"]} for i in idxs]

def rank_pages(results, return_pages=6):
    by_page = defaultdict(list)
    for r in results:
        by_page[r["page"]].append(r)
    page_scores = []
    for p, rs in by_page.items():
        s = sum(sorted([r["score"] for r in rs], reverse=True)[:3]) + 0.05*sum(len(r["text"]) for r in rs)
        page_scores.append((p, s))
    page_scores.sort(key=lambda x: -x[1])
    pages = [p for p,_ in page_scores][:return_pages]
    return pages, by_page
