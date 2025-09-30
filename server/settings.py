# server/settings.py
import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    TEXT_MODEL_PATH = os.getenv("TEXT_MODEL_PATH", "models/Qwen3-1.7B-Q4_K_M.gguf")
    MANUAL_GLOB = os.getenv("MANUAL_GLOB", "data/manuals/*.pdf")
    PAGES_DIR = os.getenv("PAGES_DIR", "data/pages")
    INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    TOP_K = int(os.getenv("TOP_K", "50"))
    RETURN_PAGES = int(os.getenv("RETURN_PAGES", "6"))
    LLM_CTX = int(os.getenv("LLM_CTX", "32768"))
    LLM_THREADS = int(os.getenv("LLM_THREADS", "8"))
    # For very large contexts, set these environment variables:
    # export LLAMA_CPP_MLOCK=1  # Lock model in RAM
    # export LLAMA_CPP_MMAP=1   # Use memory mapping
    DEFAULT_ANSWER_LANG = os.getenv("DEFAULT_ANSWER_LANG", "auto")
    TEXT_LLM_MODE = os.getenv("TEXT_LLM_MODE", "local")
    TEXT_LLM_URL = os.getenv("TEXT_LLM_URL", "http://127.0.0.1:9001")
    DEBUG = os.getenv("DEBUG", "false").lower() in ["true", "1", "yes"]

settings = Settings()


# Embeddings
setattr(Settings, 'EMBED_BACKEND', os.getenv('EMBED_BACKEND','tfidf'))
setattr(Settings, 'EMBED_MODEL_NAME', os.getenv('EMBED_MODEL_NAME','BAAI/bge-m3'))
