# server/main.py
import os, re, json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from server.settings import settings
from retriever.hybrid import load_index, hybrid_search, rank_pages
from pathlib import Path

app = FastAPI(title="Factory RAG (Monolith)")

# CORS for local UI dev (vite at :5173)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Mount static /pages for PNGs
Path(settings.PAGES_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/pages", StaticFiles(directory=settings.PAGES_DIR), name="pages")

# Serve the built UI
ui_dist = Path(__file__).resolve().parents[1] / "ui" / "web" / "dist"
if ui_dist.exists():
    # Mount static files (excluding index.html which we'll handle separately)
    app.mount("/assets", StaticFiles(directory=str(ui_dist / "assets")), name="assets")
    # Handle root path separately to serve index.html
    @app.get("/")
    async def serve_index():
        return FileResponse(str(ui_dist / "index.html"))

# Global state

g = {"llm": None, "chunks": None, "pages": None, "bm25": None, "vectorizer": None, "faiss": None}

def ensure_index():
    if g["chunks"] is None:
        if not Path(os.path.join(settings.INDEX_DIR, "manifest.json")).exists():
            raise HTTPException(500, "Index missing. Run /ingest first or build_index script.")
        g["chunks"], g["pages"], g["bm25"], g["vectorizer"], g["faiss"] = load_index(settings.INDEX_DIR)
        # Load embed_backend into global state
        import json
        with open(os.path.join(settings.INDEX_DIR, "manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        g["embed_backend"] = manifest.get("embed_backend", "tfidf")

def load_llm_local():
    from llama_cpp import Llama
    if g["llm"] is None:
        if not Path(settings.TEXT_MODEL_PATH).exists():
            raise HTTPException(500, f"Model not found: {settings.TEXT_MODEL_PATH}")
        g["llm"] = Llama(
            model_path=settings.TEXT_MODEL_PATH,
            n_ctx=settings.LLM_CTX,
            n_threads=settings.LLM_THREADS,
            logits_all=False,
            verbose=False,
            # Memory and performance optimizations for large context
            use_mlock=True,  # Lock model in memory to prevent swapping
            use_mmap=True,   # Use memory mapping for efficient loading
            # Context processing optimizations
            eval_batch_size=min(512, settings.LLM_CTX // 8),  # Adaptive batch size
            # Reduce memory fragmentation
            mul_mat_q=True,  # Use quantized matmul for Q4_K_M
        )

class AskIn(BaseModel):
    query: str
    language: str | None = None

@app.get("/health")
def health():
    ok_index = Path(os.path.join(settings.INDEX_DIR, "manifest.json")).exists()
    ok_pages = Path(settings.PAGES_DIR).exists() and len(list(Path(settings.PAGES_DIR).glob("*.png"))) > 0

    # Check SBERT encoder health if using sbert backend
    sbert_ok = True
    if ok_index:
        try:
            import json
            with open(os.path.join(settings.INDEX_DIR, "manifest.json"), "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("embed_backend") == "sbert":
                # Try to load encoder to check if it's available
                from sentence_transformers import SentenceTransformer
                SentenceTransformer(settings.EMBED_MODEL_NAME)
        except Exception:
            sbert_ok = False

    return {
        "index": ok_index,
        "pages": ok_pages,
        "llm_mode": settings.TEXT_LLM_MODE,
        "model_path": settings.TEXT_MODEL_PATH if settings.TEXT_LLM_MODE=="local" else None,
        "llm_url": settings.TEXT_LLM_URL if settings.TEXT_LLM_MODE=="remote" else None,
        "sbert_encoder": sbert_ok if ok_index else None,
    }

@app.post("/ingest")
def ingest(file: UploadFile = File(None)):
    import subprocess
    import os
    import glob
    from werkzeug.utils import secure_filename

    # Handle file upload if provided
    uploaded_file_path = None
    if file:
        # Ensure manuals directory exists
        manuals_dir = os.path.dirname(settings.MANUAL_GLOB.replace("*.pdf", ""))
        os.makedirs(manuals_dir, exist_ok=True)

        # Save uploaded file
        filename = secure_filename(file.filename)
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(400, "Only PDF files are supported")

        uploaded_file_path = os.path.join(manuals_dir, filename)

        # Save uploaded file content to disk
        try:
            with open(uploaded_file_path, "wb") as buffer:
                content = file.file.read()
                buffer.write(content)
        except Exception as e:
            raise HTTPException(500, f"Failed to save uploaded file: {str(e)}")

        # Verify file was saved
        if not os.path.exists(uploaded_file_path):
            raise HTTPException(500, f"Failed to save uploaded file: {uploaded_file_path}")

    try:
        # Check if there are any PDF files to process
        manuals_pattern = os.path.join(manuals_dir, "*.pdf")
        pdf_files = glob.glob(manuals_pattern)

        if not pdf_files:
            raise HTTPException(500, f"No PDF files found in {manuals_pattern}")

        print(f"Found PDF files: {pdf_files}")

        cmd = [
            "python","-m","retriever.build_index",
            "--manual-glob", settings.MANUAL_GLOB,
            "--pages-dir", settings.PAGES_DIR,
            "--index-dir", settings.INDEX_DIR,
            "--chunk-size", str(settings.CHUNK_SIZE),
            "--overlap", str(settings.CHUNK_OVERLAP),
        ]

        print(f"Running command: {' '.join(cmd)}")

        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout

        print(f"Subprocess return code: {cp.returncode}")
        print(f"Subprocess stdout: {cp.stdout}")
        if cp.stderr:
            print(f"Subprocess stderr: {cp.stderr}")

        if cp.returncode != 0:
            error_msg = f"Ingest failed with return code {cp.returncode}"
            if cp.stderr:
                error_msg += f": {cp.stderr}"
            else:
                error_msg += f": {cp.stdout}"
            raise HTTPException(500, error_msg)

        # Clean up uploaded file after processing
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            os.remove(uploaded_file_path)

        g.update({"chunks": None})
        ensure_index()
        return {"status": "ok", "message": "PDF uploaded and processed successfully"}

    except subprocess.TimeoutExpired:
        # Clean up uploaded file on timeout
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            os.remove(uploaded_file_path)
        raise HTTPException(500, "Ingestion timed out after 5 minutes")
    except Exception as e:
        # Clean up uploaded file on error
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            os.remove(uploaded_file_path)
        print(f"Ingestion error: {str(e)}")
        raise HTTPException(500, f"Ingest failed: {str(e)}")

SYSTEM_PROMPT = """You are a factory troubleshooting assistant with STRICT limitations:

CRITICAL RESTRICTION:
- You have NO general knowledge. You are an empty system that can ONLY read and relay information from the provided manual pages.
- NEVER use information that is not explicitly stated in the provided context.
- If information is not in the provided pages, you MUST say "This information is not available in the provided manual pages."

OUTPUT RULES (ABSOLUTELY STRICT - VIOLATIONS WILL CAUSE SYSTEM FAILURE):
- FORBIDDEN: Never write "Based on the user query", "Upon reviewing", "I need to check", "The context provides", etc.
- FORBIDDEN: Never use triple backticks (```)
- FORBIDDEN: Never explain your reasoning or thinking process
- FORBIDDEN: Never describe what you're doing or checking
- Write ONLY the direct answer to the query - nothing else
- Use ONLY information explicitly written in the provided manual context
- Use short paragraphs and bullets where helpful
- Each factual claim must include a page citation like (Manual p. X[, Y])
- Cite only the most relevant pages (max 2–3), not every page in context
- If the user query is very short or ambiguous (e.g., a single word), infer the most likely intent from the provided context and respond with a compact but meaningful summary
- End with a final line: See attached slides: p. {PAGE_LIST}.
- Never include placeholder text or instructions in your answer
"""

ANSWER_INSTRUCTION = """CRITICAL: Output ONLY the answer. NO thinking. NO reasoning. NO "Based on..." or "Upon reviewing..." phrases.

Context from manual pages (THIS IS YOUR ONLY SOURCE OF KNOWLEDGE):
{CONTEXT}

User query: {QUESTION}

STRICT OUTPUT REQUIREMENTS:
1. If the answer is NOT in the context above, write ONLY: "This information is not available in the provided manual pages."
2. If the answer IS in the context, write ONLY the direct answer using that information.
3. NEVER write about what you're checking, reviewing, or thinking.
4. NEVER use triple backticks (```)
5. NEVER explain your reasoning process

Format your response as:
- Direct answer in 1-6 sentences (from context only)
- Include (Manual p. X) citations for facts
- End with: See attached slides: p. {PAGE_LIST}.

Begin your answer immediately below:
"""

def validate_response_against_context(response, context):
    """
    Check if response contains information not present in context.
    Returns a tuple (is_valid, warning_message)
    """
    # This is a basic implementation - could be enhanced with more sophisticated checks
    
    # If response says information is not available, that's valid
    not_available_phrases = [
        "not available in the provided manual",
        "information is not available",
        "not found in the manual",
        "manual does not contain",
        "no information available"
    ]
    
    response_lower = response.lower()
    if any(phrase in response_lower for phrase in not_available_phrases):
        return True, None
    
    # Check for common general knowledge indicators that shouldn't appear
    # These terms might indicate the model is using general knowledge
    suspicious_terms = [
        "typically", "usually", "generally", "commonly", "often", 
        "standard practice", "best practice", "industry standard",
        "in my experience", "as a rule", "normally"
    ]
    
    for term in suspicious_terms:
        if term in response_lower and term not in context.lower():
            return False, f"Response may contain general knowledge (found '{term}' not in context)"
    
    return True, None

def make_context(chunks_by_page, selected_pages, max_chars_per_page=600):
    ctx_lines = []
    for p in selected_pages:
        cs = sorted(chunks_by_page[p], key=lambda r: -r["score"])[:2]
        joined = " ".join([c["text"] for c in cs])[:max_chars_per_page]
        ctx_lines.append(f"- p.{p}: {joined}")
    return "\n".join(ctx_lines)

def extract_cited_pages(text_md, default_pages):
    found = set()
    for m in re.finditer(r"\(Manual p\. ([0-9,\s\-]+)\)", text_md):
        nums = m.group(1).replace(" ", "").split(",")
        for n in nums:
            if "-" in n:
                a,b = n.split("-")
                if a.isdigit() and b.isdigit():
                    for x in range(int(a), int(b)+1):
                        found.add(x)
            else:
                if n.isdigit(): found.add(int(n))
    out = []
    for p in default_pages:
        if p in found: out.append(p)
    for p in sorted(found):
        if p not in out: out.append(p)
    return out or default_pages

def get_page_content(chunks, page_numbers):
    """Extract full text content for given page numbers from chunks."""
    page_content = {}
    for page_num in page_numbers:
        content = []
        for chunk in chunks:
            if chunk["page"] == page_num:
                content.append(chunk["text"])
        page_content[page_num] = " ".join(content)
    return page_content

def call_llm_local(prompt, stream=False):
    load_llm_local()
    if stream:
        # For streaming, we'll use a generator approach with character-level streaming
        def generate_response():
            full_text = ""
            repetition_count = 0
            last_tokens = []
            
            try:
                for token in g["llm"](
                    prompt=prompt, 
                    max_tokens=1500, 
                    temperature=0.1,  # Very low temperature for factual responses
                    top_p=0.85,
                    top_k=30,  # Reduced top_k for more conservative output
                    stop=["```", "\nAnswer:", "\n\nAnswer:"],  # Stop on code blocks and repeated answer sections
                    repeat_penalty=1.2,  # Increased repeat penalty
                    stream=True
                ):
                    text = token["choices"][0]["text"]
                    full_text += text
                    
                    # Track recent tokens to detect repetition
                    last_tokens.append(text)
                    if len(last_tokens) > 50:  # Increase window for better detection
                        last_tokens.pop(0)
                    
                    # Improved repetition detection: check for sentence-level repetition
                    if len(full_text) > 200:
                        # Check if the last 100 characters repeat earlier in the text
                        last_chunk = full_text[-100:]
                        earlier_text = full_text[:-100]
                        # Look for substantial overlap (>50 chars)
                        if len(last_chunk) > 50 and last_chunk[:50] in earlier_text:
                            repetition_count += 1
                            if repetition_count > 2:
                                print(f"Stopping generation due to sentence-level repetition")
                                break
                        else:
                            repetition_count = max(0, repetition_count - 1)
                    
                    # Stream at character level for smoother display
                    for char in text:
                        yield f"data: {json.dumps({'type': 'token', 'char': char})}\n\n"
                        
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            finally:
                yield f"data: {json.dumps({'type': 'complete', 'full_text': full_text})}\n\n"
        return generate_response()
    else:
        out = g["llm"](
            prompt=prompt, 
            max_tokens=1500, 
            temperature=0.1,  # Very low temperature for factual, document-based responses
            top_p=0.85,       # Reduced top_p to limit creativity
            top_k=30,         # Reduced top_k to be more conservative
            stop=["```", "\nAnswer:", "\n\nAnswer:"],  # Stop on code blocks and repeated answer sections
            repeat_penalty=1.2  # Increased repeat penalty
        )
        return out["choices"][0]["text"].strip()

def call_llm_remote(prompt, stream=False):
    import requests
    if stream:
        # For streaming, we'll simulate character-level streaming from remote response
        def generate_response():
            try:
                r = requests.post(f"{settings.TEXT_LLM_URL}/completion", json={
                    "prompt": prompt, "temperature": 0.1, "top_p": 0.85, "n_predict": 1500, "repeat_penalty": 1.2,
                    "stop": ["```", "\nAnswer:", "\n\nAnswer:"],
                    "stream": True
                }, timeout=120, stream=True)

                r.raise_for_status()
                full_text = ""

                for line in r.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                if 'content' in data:
                                    text = data['content']
                                    full_text += text
                                    for char in text:
                                        yield f"data: {json.dumps({'type': 'token', 'char': char})}\n\n"
                            except json.JSONDecodeError:
                                continue

                yield f"data: {json.dumps({'type': 'complete', 'full_text': full_text})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return generate_response()
    else:
        r = requests.post(f"{settings.TEXT_LLM_URL}/completion", json={
            "prompt": prompt, "temperature": 0.1, "top_p": 0.85, "n_predict": 1500, "repeat_penalty": 1.2,
            "stop": ["```", "\nAnswer:", "\n\nAnswer:"]
        }, timeout=120)
        r.raise_for_status()
        return r.json().get("content","").strip() or r.json().get("completion","").strip()

def clean_response(text):
    """Remove duplicate sentences and repetitive content from LLM response."""
    if not text:
        return text
    
    # Store original text for debugging
    original_text = text
    
    # CRITICAL: Remove code blocks with triple backticks (shouldn't be in responses)
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # Remove lines that are clearly thinking/reasoning process
    thinking_lines = [
        r'^.*?Based on the user query.*?$',
        r'^.*?Upon reviewing.*?$',
        r'^.*?I need to check.*?$',
        r'^.*?Let me (?:check|think|review).*?$',
        r'^.*?The user is asking.*?$',
        r'^.*?Since the query.*?$',
        r'^.*?the answer should.*?$',
        r'^.*?the context (?:provides|discusses|does not).*?$',
    ]
    
    for pattern in thinking_lines:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove duplicate "Answer:" prefixes that LLM might generate
    text = re.sub(r'^(Answer:\s*)+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n+(Answer:\s*)+', '\n', text, flags=re.IGNORECASE)
    
    # CRITICAL: Detect and remove repeated paragraphs (common repetition pattern)
    # Split by double newlines to get paragraphs
    paragraphs = text.split('\n\n')
    seen_paragraphs = []
    for para in paragraphs:
        para_normalized = ' '.join(para.lower().split())
        # Check if this paragraph is substantially similar to a previous one
        is_duplicate = False
        for seen_para in seen_paragraphs:
            seen_normalized = ' '.join(seen_para.lower().split())
            # If 80% of the paragraph matches a previous one, consider it a duplicate
            if len(para_normalized) > 30 and len(seen_normalized) > 30:
                if para_normalized in seen_normalized or seen_normalized in para_normalized:
                    is_duplicate = True
                    break
                # Check for high similarity (more than 80% overlap)
                min_len = min(len(para_normalized), len(seen_normalized))
                if min_len > 30:
                    overlap = sum(1 for i in range(min_len) if para_normalized[i] == seen_normalized[i])
                    if overlap / min_len > 0.8:
                        is_duplicate = True
                        break
        if not is_duplicate and para.strip():
            seen_paragraphs.append(para)
    
    text = '\n\n'.join(seen_paragraphs)
    
    # Clean up multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    # Detect and remove repetitive fragments at the end (common with token limit)
    # Check last 100 chars for repetition
    if len(text) > 150:
        last_chunk = text[-100:]
        # Look for phrases that repeat
        words = last_chunk.split()
        if len(words) > 5:
            # Check if the last few words are starting to repeat earlier words
            for i in range(3, min(15, len(words))):
                ending = ' '.join(words[-i:])
                # Check if this ending appears earlier in the last chunk
                search_text = ' '.join(words[:-i])
                if ending in search_text:
                    # Found repetition, truncate at the start of repetition
                    truncate_point = text.rfind(ending, 0, len(text) - len(ending))
                    if truncate_point > 0:
                        text = text[:truncate_point].rstrip()
                        break

    # Remove any echoed placeholder/meta lines that models sometimes produce
    # e.g., "Where {PAGE_LIST} is replaced by the relevant pages." or similar
    text = re.sub(r"^\s*Where\s+\{?PAGE_LIST\}?\s+is\s+replaced\s+by\s+the\s+relevant\s+pages\.?\s*$",
                  "",
                  text,
                  flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*Where\s+p\.?\s*\{?PAGE_LIST\}?\s*is\s+replaced.*$",
                  "",
                  text,
                  flags=re.IGNORECASE | re.MULTILINE)

    # Split into sentences (basic sentence splitting)
    sentences = []
    current_sentence = ""
    for char in text:
        current_sentence += char
        if char in '.!?' and (len(current_sentence) > 10):  # Basic sentence end detection
            sentences.append(current_sentence.strip())
            current_sentence = ""

    if current_sentence.strip() and len(current_sentence) > 20:  # Only add if substantial
        sentences.append(current_sentence.strip())

    # Remove duplicate sentences
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        # Create a normalized version for comparison (lowercase, remove extra spaces)
        normalized = ' '.join(sentence.lower().split())
        if normalized not in seen and len(sentence.strip()) > 10:  # Filter very short sentences
            seen.add(normalized)
            unique_sentences.append(sentence)

    result = ' '.join(unique_sentences).strip()

    # If result is empty (e.g., only bullets or stripped thinking), try to salvage useful content
    if not result or len(result.strip()) < 10:
        # First, try to find the core answer (often starts with "The information is not available" or similar)
        not_available_match = re.search(
            r'(The information is not available.*?(?:pages?|manual)[^.]*\.)', 
            original_text, 
            flags=re.IGNORECASE
        )
        if not_available_match:
            result = not_available_match.group(1).strip()
        else:
            # Try to salvage bullet lines
            bullet_lines = []
            for line in original_text.splitlines():
                line_stripped = line.strip()
                if line_stripped.startswith(('-', '*', '•')) and len(line_stripped) > 2:
                    bullet_lines.append(line_stripped)
            if bullet_lines:
                result = '\n'.join(bullet_lines[:4]).strip()
    
    # Special handling: If we detect "See attached slides" at the beginning but removed everything else
    if re.match(r'^See attached slides:', result, flags=re.IGNORECASE) and len(result) < 50:
        # Try to find the actual answer in the original text
        not_available_match = re.search(
            r'((?:This|The) information is not available.*?(?:pages?|manual)[^.]*\.)',
            original_text,
            flags=re.IGNORECASE
        )
        if not_available_match:
            result = not_available_match.group(1).strip()
    
    # Log when significant content is removed (for debugging)
    if len(result) < len(original_text) * 0.3:  # If more than 70% was removed
        print(f"Warning: clean_response removed significant content")
        print(f"Original length: {len(original_text)}, Final length: {len(result)}")
        if settings.DEBUG:
            print(f"Original text preview: {original_text[:500]}...")
            print(f"Cleaned text: {result}")
    
    # If cleaning removed too much content, try to extract the essential answer
    if len(result.strip()) < 20 and len(original_text.strip()) > 50:
        print(f"Warning: clean_response removed too much content, attempting recovery")
        # Look for key answer patterns
        answer_patterns = [
            r'((?:This|The) information is not available[^.]*\.)',
            r'((?:According to|Based on|From) (?:the )?(?:manual|page)[^.]*\.)',
            r'(The [\w\s]+ (?:is|are|indicates?|shows?|requires?)[^.]*\.)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, original_text, flags=re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                print(f"Recovered answer: {result}")
                break
        
        if len(result.strip()) < 20:  # Still too short
            # Last resort: minimal cleanup of original
            result = re.sub(r'^(Answer:\s*)+', '', original_text, flags=re.IGNORECASE)
            result = re.sub(r'\n+(Answer:\s*)+', '\n', result, flags=re.IGNORECASE)
            result = re.sub(r'```[\s\S]*?```', '', result)  # Remove code blocks
            result = result.strip()
    
    return result

@app.post("/ask")
def ask(body: AskIn):
    # For backward compatibility, we'll use the streaming endpoint internally
    # and return the complete response
    ensure_index()
    query = body.query.strip()
    if not query:
        raise HTTPException(400, "Empty query.")

    # Language handling
    if settings.DEFAULT_ANSWER_LANG == "auto":
        try:
            from langdetect import detect
            qlang = body.language or detect(query)
        except Exception:
            qlang = body.language or "en"
    else:
        qlang = settings.DEFAULT_ANSWER_LANG

    embed_backend = g.get('embed_backend') or 'tfidf'
    enc = encoder_fn if embed_backend=='sbert' else None
    hits = hybrid_search(query, g["chunks"], g["bm25"], g["vectorizer"], g["faiss"], embed_backend=embed_backend, encoder=enc, top_k=settings.TOP_K)
    selected_pages, by_page = rank_pages(hits, settings.RETURN_PAGES)

    ctx = make_context(by_page, selected_pages)
    page_list = ", ".join(str(p) for p in selected_pages)
    prompt = f"{SYSTEM_PROMPT}\n\n" + ANSWER_INSTRUCTION.format(
        CONTEXT=ctx, QUESTION=query, PAGE_LIST=page_list
    )
    if qlang and qlang != "en":
        prompt = f"Answer in {qlang}.\n" + prompt

    # Get the complete response
    raw_text = call_llm_local(prompt, stream=False) if settings.TEXT_LLM_MODE=="local" else call_llm_remote(prompt, stream=False)
    
    # Log raw response for debugging
    if settings.DEBUG or len(raw_text.strip()) < 50:
        print(f"Raw LLM response (length: {len(raw_text)}): {raw_text[:500]}...")
    
    text = clean_response(raw_text)
    
    # Validate that response doesn't contain general knowledge
    is_valid, warning = validate_response_against_context(text, ctx)
    if not is_valid and warning:
        print(f"WARNING: {warning}")
        if settings.DEBUG:
            print(f"Response that triggered warning: {text[:200]}...")
    
    # Ensure we have a complete response
    if not text or len(text.strip()) < 30:  # If response is too short, it might be truncated
        print(f"Warning: LLM response may be incomplete after cleaning.")
        print(f"Raw length: {len(raw_text)}, Cleaned length: {len(text) if text else 0}")
        print(f"Cleaned response: {text}")
        # If cleaning removed everything, use raw text with minimal cleanup
        if len(raw_text.strip()) > len(text.strip()) * 2:
            print("Using raw text due to aggressive cleaning")
            text = re.sub(r'^(Answer:\s*)+', '', raw_text, flags=re.IGNORECASE).strip()

    # Final guard: if empty or only punctuation/whitespace, provide standardized fallback
    if not text or not re.search(r"[A-Za-z0-9]", text):
        text = "This information is not available in the provided manual pages."

    # Extract and limit citations to the most relevant (top 3)
    cited = extract_cited_pages(text, selected_pages)[:3]

    # Normalize final slide line to use the limited citations
    final_page_list = ", ".join(str(p) for p in cited) if cited else page_list
    # Remove any existing 'See attached slides' lines and placeholders, then append the normalized line
    text = re.sub(r"\n?See attached slides: p\.[^\n]*\.?\s*$", "", text, flags=re.IGNORECASE)
    text = text.replace("{PAGE_LIST}", final_page_list)
    if not text.rstrip().endswith("."):
        text = text.rstrip()
    text = text.rstrip() + f"\nSee attached slides: p. {final_page_list}."

    slides = [{"page": p, "image_url": f"/pages/page_{p:03d}.png", "alt": f"Manual p.{p}"} for p in cited]
    citations = [{"page": p, "snippet": ""} for p in cited]
    page_content = get_page_content(g["chunks"], cited)

    return JSONResponse({
        "language": qlang,
        "answer_md": text,
        "citations": citations,
        "slides": slides,
        "page_content": page_content
    })

@app.post("/ask/stream")
def ask_stream(body: AskIn):
    ensure_index()
    query = body.query.strip()
    if not query:
        raise HTTPException(400, "Empty query.")

    # Language handling
    if settings.DEFAULT_ANSWER_LANG == "auto":
        try:
            from langdetect import detect
            qlang = body.language or detect(query)
        except Exception:
            qlang = body.language or "en"
    else:
        qlang = settings.DEFAULT_ANSWER_LANG

    embed_backend = g.get('embed_backend') or 'tfidf'
    enc = encoder_fn if embed_backend=='sbert' else None
    hits = hybrid_search(query, g["chunks"], g["bm25"], g["vectorizer"], g["faiss"], embed_backend=embed_backend, encoder=enc, top_k=settings.TOP_K)
    selected_pages, by_page = rank_pages(hits, settings.RETURN_PAGES)

    ctx = make_context(by_page, selected_pages)
    page_list = ", ".join(str(p) for p in selected_pages)
    prompt = f"{SYSTEM_PROMPT}\n\n" + ANSWER_INSTRUCTION.format(
        CONTEXT=ctx, QUESTION=query, PAGE_LIST=page_list
    )
    if qlang and qlang != "en":
        prompt = f"Answer in {qlang}.\n" + prompt

    def generate_response():
        try:
            # Send initial metadata
            yield f"data: {json.dumps({'type': 'metadata', 'language': qlang, 'pages': selected_pages})}\n\n"

            # Stream LLM response
            full_text = ""
            if settings.TEXT_LLM_MODE == "local":
                for token_data in call_llm_local(prompt, stream=True):
                    if token_data.startswith("data: "):
                        try:
                            data = json.loads(token_data[6:])
                            if data.get('type') == 'token' and data.get('char'):
                                full_text += data['char']
                                yield token_data
                            elif data.get('type') == 'complete':
                                full_text = data.get('full_text', full_text)
                                break
                            elif data.get('type') == 'error':
                                yield token_data
                                return
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue
            else:
                # For remote LLM, use streaming if available
                for token_data in call_llm_remote(prompt, stream=True):
                    if token_data.startswith("data: "):
                        try:
                            data = json.loads(token_data[6:])
                            if data.get('type') == 'token' and data.get('char'):
                                full_text += data['char']
                                yield token_data
                            elif data.get('type') == 'complete':
                                full_text = data.get('full_text', full_text)
                                break
                            elif data.get('type') == 'error':
                                yield token_data
                                return
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue

            # Store raw text for debugging
            raw_text = full_text
            
            # Clean the full response before sending final metadata
            full_text = clean_response(full_text)
            
            # Validate that response doesn't contain general knowledge
            is_valid, warning = validate_response_against_context(full_text, ctx)
            if not is_valid and warning:
                print(f"WARNING (streaming): {warning}")
                if settings.DEBUG:
                    print(f"Response that triggered warning: {full_text[:200]}...")
            
            # Log if cleaning removed significant content
            if settings.DEBUG or len(full_text.strip()) < len(raw_text.strip()) * 0.5:
                print(f"Streaming: Raw response length: {len(raw_text)}, Cleaned: {len(full_text)}")
                if len(full_text.strip()) < 30 and len(raw_text.strip()) > 50:
                    print("Warning: Streaming response heavily cleaned, using raw text")
                    full_text = re.sub(r'^(Answer:\s*)+', '', raw_text, flags=re.IGNORECASE).strip()
            
            # Replace {PAGE_LIST} placeholder if LLM didn't replace it
            full_text = full_text.replace("{PAGE_LIST}", page_list)

            # Final guard for empty or punctuation-only content
            if not full_text or not re.search(r"[A-Za-z0-9]", full_text):
                full_text = "This information is not available in the provided manual pages."

            # Check if response seems complete (relaxed heuristic)
            # Consider complete if it either has proper ending punctuation OR includes the final slides line placeholder.
            # Allow shorter answers (>= 20 chars) so terse queries don't trigger false warnings.
            trimmed = full_text.strip()
            is_complete = (
                len(trimmed) >= 20 and (
                    trimmed.endswith(('.', '!', '?')) or 'See attached slides' in trimmed
                ) and not trimmed.endswith(('...', '..', ' a', ' the', ' is', ' and'))
            )

            # Send final data (limit citations to the most relevant top 3)
            cited = extract_cited_pages(full_text, selected_pages)[:3]
            final_page_list = ", ".join(str(p) for p in cited) if cited else ", ".join(str(p) for p in selected_pages)
            
            # Normalize final answer footer
            full_text = re.sub(r"\n?See attached slides: p\.[^\n]*\.?\s*$", "", full_text, flags=re.IGNORECASE)
            full_text = full_text.replace("{PAGE_LIST}", final_page_list)
            if not full_text.rstrip().endswith("."):
                full_text = full_text.rstrip() + "."
            full_text = full_text + f"\nSee attached slides: p. {final_page_list}."

            slides = [{"page": p, "image_url": f"/pages/page_{p:03d}.png", "alt": f"Manual p.{p}"} for p in cited]
            citations = [{"page": p, "snippet": ""} for p in cited]
            page_content = get_page_content(g["chunks"], cited)

            yield f"data: {json.dumps({'type': 'complete', 'answer_md': full_text, 'citations': citations, 'slides': slides, 'page_content': page_content, 'is_complete': is_complete})}\n\n"

        except Exception as e:
            # Send error to client
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering if behind proxy
        }
    )


# Optional SBERT encoder (for EMBED_BACKEND=sbert)
def encoder_fn(texts):
    # Lazy load sentence-transformers only if needed
    from sentence_transformers import SentenceTransformer
    model_name = settings.EMBED_MODEL_NAME
    if 'encoder_model' not in g or g.get('encoder_name') != model_name:
        g['encoder_model'] = SentenceTransformer(model_name)
        g['encoder_name'] = model_name
    vecs = g['encoder_model'].encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype('float32')
