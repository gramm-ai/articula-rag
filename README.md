# Factory RAG — Monolith Starter (with Optional BGE & UI)

RAG app for factory equipment manuals.  
Runs offline, cites exact manual pages, and returns the matching page images.

---

---

## Goals
- **Offline-first**: Run entirely on a factory floor PC or edge box.
- **Trustworthy answers**: Every claim cites **(Manual p. X)**.
- **Slide fidelity**: Return the **exact manual page image(s)** with each answer.
- **Multilingual**: Works for multilingual questions; answers in the user’s language.
- **Low setup**: `pip install`, run, upload PDF from the UI. Can be containerized later.

---

## Key Features
- **Monolith FastAPI app**:
  - `/ask` returns `answer_md`, `citations[]`, `slides[]` with image URLs, and `page_content{}` (full text of cited pages).
  - `/ask/stream` provides **real-time streaming responses** via Server-Sent Events.
  - `/ingest` builds the index (PDF → page PNGs + hybrid index) and accepts PDF upload via `multipart/form-data`.
  - `/health` reports readiness signals.
  - Serves **page PNGs** at `/pages/page_XXX.png`.
  - Optionally serves a **built React UI** at `/`.
- **Hybrid retrieval**:
  - **Baseline**: TF‑IDF vectors + **BM25** (fully offline).
  - **Optional**: **SBERT/BGE** (`BAAI/bge-m3`) via sentence-transformers for better recall & multilingual.
  - **Score fusion**: 0.60 × vector (TF‑IDF or SBERT) + 0.40 × BM25.
- **LLM**:
  - CPU‑friendly **Qwen3-1.7B** (quantized GGUF) via `llama-cpp-python` **in-process**.
  - Later: one switch to call a **remote LLM server** (same API & UI).
- **Real-time streaming**: Responses appear character-by-character as they're generated.
- **Multilingual answers**: auto language detection; technical tokens preserved verbatim.

---

## How it works (short)
- Ingest the PDF → split pages → extract text → build BM25 + FAISS index.
- `/ask` retrieves top pages and calls the text model to produce an answer with page citations.
- Response includes page images and full text for cited pages (`page_content`).
- **Real-time streaming**: `/ask/stream` provides responses as they're generated via Server-Sent Events.

---

## Real-time Streaming
The application supports **real-time response streaming** via the `/ask/stream` endpoint:

- **Server-Sent Events (SSE)**: Uses the EventSource API for efficient streaming
- **Character-by-character updates**: See responses appear as they're generated
- **Visual feedback**: Animated cursor indicates active generation
- **Complete citations**: Final response includes page citations and slide images

The web UI displays streaming responses by default, providing immediate feedback during LLM generation.

---

## Data & Response Contracts
### Request → `/ingest` (POST, multipart/form-data)
Upload a PDF to ingest. The typical flow is to upload from the UI, which sends a multipart/form-data request to this endpoint.

```bash
# macOS/Linux
curl -s -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/Manual.pdf"

# Windows (PowerShell or CMD)
curl -s -X POST http://localhost:8000/ingest -F "file=@C:\\path\\to\\Manual.pdf"
```

Response
```json
{
  "status": "ok",
  "message": "PDF uploaded and processed successfully"
}
```

### Request → `/ask` (POST)
```json
{
  "query": "How do I recover M_ERR_SYSTEM_OPEN and where are the test logs?",
  "language": null
}
```

### Response (always)
```json
{
  "language": "en",
  "answer_md": "Markdown answer ... (Manual p. 12, 14). See attached slides: p. 12, 14.",
  "citations": [
    {"page": 12, "snippet": ""},
    {"page": 14, "snippet": ""}
  ],
  "slides": [
    {"page": 12, "image_url": "/pages/page_012.png", "alt": "Manual p.12"},
    {"page": 14, "image_url": "/pages/page_014.png", "alt": "Manual p.14"}
  ],
  "page_content": {
    "12": "... full text content of page 12 ...",
    "14": "... full text content of page 14 ..."
  }
}
```

> The API **does not embed images**—it returns URLs to the exact page PNGs, ensuring **fidelity** to the manual.
>
> Additionally, the response includes `page_content{}` mapping cited page numbers to their full text so the UI can render exact manual text, enable verification, or implement highlighting.

---

## Ingestion Pipeline
Script: `python -m retriever.build_index`

1. **PDF → page PNGs** (`data/pages/page_XXX.png`) via **PyMuPDF** (150–200 DPI).
2. **Text extraction** per page (layout‑ordered).
3. **Chunking** per page (size ~700, overlap ~120).
4. **Indexes**:
   - **BM25** (lexical) on chunks with `rank-bm25`.
   - **Dense**:
     - **TF‑IDF** (default): vectorizer → normalized array → FAISS `IndexFlatIP`.
     - **SBERT/BGE** (optional): `SentenceTransformer` → normalized embeddings → FAISS.
5. **Artifacts** (`data/index/`):
   - `manifest.json` — chunk & page metadata, chosen embed backend.
   - `bm25.pkl` — BM25 model.
   - `tfidf.pkl` — TF‑IDF vectorizer (always saved; used for query features/fallback).
   - `tfidf.faiss` **or** `dense.faiss`.
   - Note: `manifest.json` includes `embed_backend` ("tfidf" or "sbert"). The server reads this to select retrieval mode; `/health` only checks the SBERT encoder when `embed_backend` is `sbert`.

Re‑ingest when you add/change manuals.

---

## Upload & Ingest Workflow
1. Start the API
   ```bash
   uvicorn server.main:app --host 0.0.0.0 --port 8000
   ```
2. Start the UI (dev)
   ```bash
   cd ui/web
   npm install
   npm run dev   # opens http://localhost:5173
   ```
3. In the UI
   - Set the API URL to your server (e.g., http://localhost:8000)
   - Use "Upload & Process PDF" to send your manual to `/ingest`
4. Verify readiness
   ```bash
   curl -s http://localhost:8000/health | jq
   # Expect: { "index": true, "pages": true, ... }
   ```
5. Ask questions against the ingested manual
   ```bash
   curl -s http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query":"What is the safety procedure for XYZ?"}' | jq
   ```

Alternative: upload via CLI instead of the UI
```bash
# macOS/Linux
curl -s -X POST http://localhost:8000/ingest -F "file=@/path/to/Manual.pdf"

# Windows
curl -s -X POST http://localhost:8000/ingest -F "file=@C:\\path\\to\\Manual.pdf"
```

Notes
- PDFs are saved under `data/manuals/` automatically by the server. You do not need to place files there manually.
- Page images are served from `/pages/page_XXX.png` after ingestion.
- You can upload multiple PDFs (via UI or CLI). Re-run `/ingest` whenever manuals change.

---

## Retrieval
- **Default**: Hybrid BM25 + TF‑IDF FAISS.
- **Optional**: Hybrid BM25 + SBERT/BGE FAISS (`EMBED_BACKEND=sbert`, `EMBED_MODEL_NAME=BAAI/bge-m3`).
- **Fusion**: score = 0.60 × vector (TF‑IDF or SBERT) + 0.40 × BM25; top `TOP_K` hits are considered, then pages are ranked and the top `RETURN_PAGES` are used for context and slides.

---

## Generation & citations
- Uses `llama-cpp-python` (local) or a remote server.
- Answers must include page citations `(Manual p. X[,Y])`.
- Images are returned as URLs; no embedding in markdown.

---

## UI
Located at `ui/web/` (Vite + React):
- **Dev mode**: `npm run dev` at <http://localhost:5173>, API at <http://localhost:8000> (CORS enabled).
- **Build**: `npm run build` → `ui/web/dist/`. The API will **serve `/`** from this folder if present.

UI renders:
- `answer_md` (plain text for now),
- `citations` as page numbers,
- a right‑rail of **Relevant slides** (actual PNGs from `/pages/...`).
- optional: `page_content` as the full text of cited pages (for inline text view and highlighting).

---

## Configuration (.env)
Copy `.env.example` → `.env`, adjust as needed.

```ini
# Paths
TEXT_MODEL_PATH=models/Qwen3-1.7B-Q4_K_M.gguf
MANUAL_GLOB=data/manuals/*.pdf
PAGES_DIR=data/pages
INDEX_DIR=data/index

# Retrieval
CHUNK_SIZE=700
CHUNK_OVERLAP=120
TOP_K=50
RETURN_PAGES=6

# LLM (text)
LLM_CTX=8192
LLM_THREADS=8
TEXT_LLM_MODE=local           # or 'remote'
TEXT_LLM_URL=http://127.0.0.1:9001

# Embeddings
EMBED_BACKEND=tfidf           # or 'sbert'
EMBED_MODEL_NAME=BAAI/bge-m3  # used when EMBED_BACKEND=sbert

# Locale
DEFAULT_ANSWER_LANG=auto
```

---

## Quickstart
```bash
# 1) Python env
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install -r requirements.txt

# 2) Configure
cp .env.example .env

# 3) Model
# Put your quantized text model under models/ (e.g. Qwen3-1.7B-Q4_K_M.gguf)

# Download the model file (choose one):

Linux/macOS (curl)
```bash
mkdir -p models
curl -L \
  -o models/Qwen3-1.7B-Q4_K_M.gguf \
  "<DIRECT_URL_TO_GGUF_FILE>"

# Optional: using Hugging Face CLI
pip install -U "huggingface_hub[cli]"
huggingface-cli download <REPO_ID> Qwen3-1.7B-Q4_K_M.gguf --local-dir models
```

Windows (PowerShell)
```powershell
New-Item -ItemType Directory -Force -Path models | Out-Null
$ModelUrl = "<DIRECT_URL_TO_GGUF_FILE>"
Invoke-WebRequest -Uri $ModelUrl -OutFile "models\Qwen3-1.7B-Q4_K_M.gguf"

# Optional: using Hugging Face CLI
pip install "huggingface_hub[cli]"
huggingface-cli download <REPO_ID> Qwen3-1.7B-Q4_K_M.gguf --local-dir models
```

Then set or confirm `TEXT_MODEL_PATH` in `.env` (default: `models/Qwen3-1.7B-Q4_K_M.gguf`).

# 4) Run API
uvicorn server.main:app --host 0.0.0.0 --port 8000

# 5) Start the UI (dev) and upload your PDF
cd ui/web
npm install
npm run dev   # open http://localhost:5173
# In the UI, set API URL to http://localhost:8000 and click "Upload & Process PDF"

# 6) Ask
curl -s http://localhost:8000/health | jq
curl -s http://localhost:8000/ask -H "Content-Type: application/json"   -d '{"query":"How do I recover M_ERR_SYSTEM_OPEN and where are the test logs?"}' | jq
# open http://localhost:8000/pages/page_001.png
```

<!-- Manual ingest script removed: ingestion is driven via UI upload or /ingest endpoint -->

---

## Optional: multilingual embeddings
```bash
pip install sentence-transformers==3.0.1
# .env
EMBED_BACKEND=sbert
EMBED_MODEL_NAME=BAAI/bge-m3
python -m retriever.build_index
```

---

## Optional: remote LLM
```ini
# .env
TEXT_LLM_MODE=remote
TEXT_LLM_URL=http://127.0.0.1:9001
```
Run llama.cpp server with your model.

---

## Health
- `GET /health` — readiness flags (index + pages + LLM mode).
 - When the index was built with SBERT (`embed_backend: "sbert"`), the endpoint also verifies the SBERT model (`EMBED_MODEL_NAME`) can be loaded.

---

## Testing
- `/health` should return `index: true` and `pages: true`.
- `/ask` should include `(Manual p. X)` citations, non‑empty `slides[]`, and `page_content{}` for cited pages.

---

## Notes
- CPU-only works. Increase `LLM_THREADS` to match your cores.
- Keep `LLM_CTX` around 8k unless you know you need more.

---

## Offline
- No cloud calls after models are present.

---

## Document-Only Responses

The system is configured to provide responses **strictly from the uploaded documents**. The LLM will:
- Only use information explicitly present in the manual pages
- Reject queries that require general knowledge not in the documents
- Include page citations for all factual claims
- Use very low temperature (0.1) to minimize creativity

To enable debug logging for response validation:
```bash
export DEBUG=true  # or DEBUG=1
```

## Troubleshooting
- **Upload to /ingest fails**: confirm content type is `multipart/form-data` and the field name is `file`.
- **No PDFs found during ingest**: ensure you selected a PDF in the UI and retried.
- **Ingestion timed out**: large PDFs may take longer; retry the upload or try the CLI upload example above.
- **/health shows pages=false**: run ingestion (upload a PDF); ensure PNGs exist in `data/pages/`.
- **Model not found**: check `TEXT_MODEL_PATH`.
- **SBERT errors**: set `EMBED_BACKEND=tfidf` or install `sentence-transformers`.
- **CORS issues with the UI**: API enables `allow_origins="*"`; confirm the API port in the UI.
- **LLM using general knowledge**: Check console for warnings about general knowledge indicators.

---

<!-- Roadmap removed to keep this concise. -->

---

## License
MIT.
