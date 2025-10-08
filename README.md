# Factory RAG System

> **Production-ready Retrieval-Augmented Generation (RAG) for factory equipment manuals**  
> Runs fully offline with precise page citations and visual references.

## Key Features

- **Offline-first** - Runs entirely on local hardware, no cloud dependencies
- **Precise citations** - Every answer includes exact manual page references
- **Visual fidelity** - Returns original page images alongside text responses  
- **Real-time streaming** - Character-by-character response generation
- **Multilingual support** - Auto-detects language and preserves technical terms

## Quick Start

### Prerequisites

- Python 3.8+ (3.10+ recommended)
- Node.js 16+ (for UI)
- 8GB+ RAM
- 10GB+ disk space

### Installation

1. **Set up Python environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Download the model**

Linux/macOS:
```bash
mkdir -p models
curl -L -o models/Qwen3-1.7B-Q4_K_M.gguf "<DIRECT_URL_TO_GGUF_FILE>"
```

Windows PowerShell:
```powershell
New-Item -ItemType Directory -Force -Path models
Invoke-WebRequest -Uri "<DIRECT_URL_TO_GGUF_FILE>" -OutFile "models\Qwen3-1.7B-Q4_K_M.gguf"
```

3. **Configure environment**

Create `.env` file:
```ini
TEXT_MODEL_PATH=models/Qwen3-1.7B-Q4_K_M.gguf
MANUAL_GLOB=data/manuals/*.pdf
PAGES_DIR=data/pages
INDEX_DIR=data/index
CHUNK_SIZE=700
CHUNK_OVERLAP=120
TOP_K=50
RETURN_PAGES=6
LLM_CTX=8192
LLM_THREADS=8
TEXT_LLM_MODE=local
EMBED_BACKEND=tfidf
DEFAULT_ANSWER_LANG=auto
```

4. **Start the services**
```bash
# Terminal 1: Start API
uvicorn server.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start UI (development)
cd ui/web
npm install
npm run dev
```

5. **Upload and query**
- Open http://localhost:5173
- Upload a PDF manual via the UI
- Ask questions about the content

## Project Structure

```
articula-rag/
├── server/              # FastAPI backend
│   ├── main.py         # API endpoints
│   └── settings.py     # Configuration
├── retriever/           # Search & indexing
│   ├── build_index.py  # PDF processing
│   └── hybrid.py       # Hybrid search
├── ui/web/             # React frontend
├── models/             # GGUF model files
└── data/               # Runtime data
    ├── manuals/        # PDF uploads
    ├── pages/          # Page images
    └── index/          # Search indexes
```

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System readiness check |
| `/ingest` | POST | Upload and process PDF |
| `/ask` | POST | Query the system |
| `/ask/stream` | POST | Stream responses in real-time |
| `/pages/*.png` | GET | Serve page images |

### Request/Response Examples

**Upload PDF:**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/manual.pdf"
```

**Ask Question:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I fix M_ERR_SYSTEM_OPEN?"}'
```

**Response Format:**
```json
{
  "answer_md": "The solution is... (Manual p. 12)",
  "citations": [{"page": 12, "snippet": ""}],
  "slides": [{"page": 12, "image_url": "/pages/page_012.png"}],
  "page_content": {"12": "Full text of page 12..."}
}
```

## Configuration Options

### Search Methods

- **Default**: TF-IDF + BM25 (offline, fast)
- **Enhanced**: SBERT/BGE embeddings (better multilingual)

To enable SBERT:
```bash
pip install sentence-transformers==3.0.1
# In .env:
EMBED_BACKEND=sbert
EMBED_MODEL_NAME=BAAI/bge-m3
```

### Remote LLM

To use an external LLM server:
```ini
# In .env:
TEXT_LLM_MODE=remote
TEXT_LLM_URL=http://127.0.0.1:9001
```

### Debug Mode

Enable detailed logging:
```bash
export DEBUG=true  # Linux/macOS
$env:DEBUG="true"  # Windows PowerShell
```

## How It Works

1. **Ingestion**: PDF → Page images + Text extraction → Chunking → Index building
2. **Retrieval**: Hybrid search (BM25 + Dense vectors) → Top-K page ranking
3. **Generation**: Context + Query → LLM → Cited answer with page references
4. **Response**: Markdown text + Citations + Page images + Full page content

### Key Components

- **FastAPI** backend with SSE streaming support
- **React** UI with real-time response display
- **FAISS** for vector similarity search
- **BM25** for lexical search
- **llama-cpp-python** for local LLM inference

## Important Notes

### Document-Only Responses
The system only uses information from uploaded documents:
- No general knowledge responses
- All claims require page citations
- Low temperature (0.1) for factual accuracy

### Performance Tips
- Increase `LLM_THREADS` to match CPU cores
- Keep `LLM_CTX` at 8192 unless needed
- For large PDFs, ingestion may take several minutes

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Upload fails | Check file is PDF, verify server is running |
| No pages found | Re-upload PDF, check `data/pages/` directory |
| Model not found | Verify `TEXT_MODEL_PATH` in `.env` |
| Slow responses | Increase `LLM_THREADS`, check CPU usage |
| CORS errors | Ensure API URL matches in UI settings |

## License

MIT License