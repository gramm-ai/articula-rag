# retriever/build_index.py (improved with optional SBERT embeddings)
import os, glob, json
import argparse
import fitz
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

def rasterize_pdf(pdf_path, pages_dir, dpi=180):
    try:
        os.makedirs(pages_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        images = []
        for i in range(len(doc)):
            try:
                page = doc[i]
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                fname = f"page_{i+1:03d}.png"
                out = os.path.join(pages_dir, fname)
                pix.save(out)
                images.append({"page": i+1, "file": fname})
            except Exception as e:
                print(f"Warning: Failed to process page {i+1} of {pdf_path}: {e}")
                continue
        doc.close()
        return images
    except Exception as e:
        raise SystemExit(f"Error processing PDF {pdf_path}: {e}")

def chunk_page(text, page_no, chunk_size=700, overlap=120):
    toks = text.split()
    chunks = []
    start = 0
    while start < len(toks):
        end = min(len(toks), start + chunk_size)
        chunk_txt = " ".join(toks[start:end]).strip()
        if chunk_txt:
            chunks.append({"page": page_no, "text": chunk_txt})
        if end == len(toks): break
        start = max(0, end - overlap)
    return chunks

def extract_texts(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        texts = []
        for i in range(len(doc)):
            try:
                page = doc[i]
                text = page.get_text("text", sort=True)
                texts.append((i+1, text))
            except Exception as e:
                print(f"Warning: Failed to extract text from page {i+1} of {pdf_path}: {e}")
                texts.append((i+1, ""))
                continue
        doc.close()
        return texts
    except Exception as e:
        raise SystemExit(f"Error extracting text from PDF {pdf_path}: {e}")

def build_tfidf(chunks):
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=100_000)
    X = vectorizer.fit_transform([c["text"] for c in chunks])
    X = X.astype(np.float32).toarray()
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    return vectorizer, X

def build_sbert(chunks, model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise SystemExit(f"EMBED_BACKEND=sbert requires sentence-transformers. Install it and retry. ({e})")
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    if texts:
        X = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    else:
        X = np.zeros((0, 384), dtype=np.float32)
    return {"model_name": model_name}, X

def save_faiss(X, index_dir, filename):
    os.makedirs(index_dir, exist_ok=True)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    if X.shape[0] > 0:
        index.add(X)
    faiss.write_index(index, os.path.join(index_dir, filename))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual-glob", default=os.getenv("MANUAL_GLOB", "data/manuals/*.pdf"))
    ap.add_argument("--pages-dir", default=os.getenv("PAGES_DIR", "data/pages"))
    ap.add_argument("--index-dir", default=os.getenv("INDEX_DIR", "data/index"))
    ap.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "700")))
    ap.add_argument("--overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "120")))
    ap.add_argument("--if-missing", action="store_true")
    # embeddings
    EMBED_BACKEND = os.getenv("EMBED_BACKEND", "tfidf").lower()
    EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
    args = ap.parse_args()

    # Validate parameters
    if not args.manual_glob:
        raise SystemExit("Error: MANUAL_GLOB cannot be empty")
    if args.chunk_size <= 0:
        raise SystemExit("Error: chunk-size must be positive")
    if args.overlap < 0:
        raise SystemExit("Error: overlap must be non-negative")
    if args.overlap >= args.chunk_size:
        raise SystemExit("Error: overlap must be less than chunk-size")

    Path(args.pages_dir).mkdir(parents=True, exist_ok=True)
    Path(args.index_dir).mkdir(parents=True, exist_ok=True)

    if args.if_missing and Path(os.path.join(args.index_dir, "manifest.json")).exists():
        print("Index exists; skipping.")
        return

    pdfs = sorted(glob.glob(args.manual_glob))
    if not pdfs:
        raise SystemExit(f"No PDFs found at {args.manual_glob}")

    all_chunks = []
    page_images = []
    for pdf in pdfs:
        imgs = rasterize_pdf(pdf, args.pages_dir)
        page_images.extend(imgs)
        for pno, ptxt in extract_texts(pdf):
            all_chunks.extend(chunk_page(ptxt, pno, args.chunk_size, args.overlap))

    tokenized = [c["text"].split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized) if tokenized else BM25Okapi([[]])

    meta = {
        "chunks": all_chunks,
        "pages": page_images,
        "embed_backend": EMBED_BACKEND,
    }

    import pickle
    if EMBED_BACKEND == "sbert":
        info, X = build_sbert(all_chunks, EMBED_MODEL_NAME)
        meta["sbert_info"] = info
        save_faiss(X, args.index_dir, "dense.faiss")
        # still save a tiny TF-IDF for fallback query vectorization if needed?
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,1), max_features=1_000)
        vectorizer.fit([c["text"] for c in all_chunks] or [""])
        with open(os.path.join(args.index_dir, "tfidf.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)
    else:
        vectorizer, X = build_tfidf(all_chunks)
        with open(os.path.join(args.index_dir, "tfidf.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)
        save_faiss(X, args.index_dir, "tfidf.faiss")

    with open(os.path.join(args.index_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    with open(os.path.join(args.index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    print("Index built with backend:", EMBED_BACKEND)

if __name__ == "__main__":
    main()
