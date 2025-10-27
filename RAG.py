import os, json, time
from pathlib import Path
import numpy as np
import streamlit as st
from pypdf import PdfReader                             # Basic PDF text extraction
import fitz                                             # PyMuPDF more accurate text extraction with layout awareness
from sentence_transformers import SentenceTransformer
import faiss                                            #(Facebook AI Similarity Search)
import re

import kpi_logger                                       # Custom module for performance logging
from datetime import datetime

# ---- Starting log app for performance tracking ----
kpi_logger.log_metrics(
    query="__app_start__",
    response_latency=0.0,
    retrieval_latency=0.0,
    inter_chunk_sim=0.0
)
print(f"[{datetime.now().isoformat()}] Metrics logger initialized.")


# ---- Directory SetUp and Constants ----

RAW_DIR = Path("data/raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)          # Directory for PDF/TXT files
IDX_DIR = Path("data/index"); IDX_DIR.mkdir(parents=True, exist_ok=True)        # Directory for indexed file
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"                       # Pre-trained Embedding model
TEXTS_PATH = IDX_DIR / "texts.jsonl"                                            # Directory for text chunks
INDEX_PATH = IDX_DIR / "faiss.index"                                            # Directory for FAISS index
USE_LOCAL_LLM = True                                                            # Whether to use a local model for summarization

# Regex to detect paragraph markers from PyMuPDF output
P_MARK = re.compile(r"^---\s*<P:(\d+)\.(\d+)>\s*---\s*$")

# Minimum number of words required to store a text chunk
MIN_CHUNK_WORDS = 10


# ---- File loading ----

def chunk_text(text: str, size=200, overlap=0.2):
    """
    Split text into overlapping chunks of roughly `size` words.
    in evaluate_chunk_sizes_on_data it is tested with different chunk sizes.
    in Build index: the chosen size will be used
    Overlap is proportional to chunk size: 20%
    """
    words = text.split()
    size = int(max(1, round(size)))
    step = int(max(1, round(size * (1 - overlap))))  # compute overlap step e.g. 20% Overlap
    out, i, n = [], 0, len(words)

    # Loop through text and generate chunks
    while i < n:
        j = min(n, i + size)
        out.append(" ".join(words[i:j]))
        i += step
    return out

def extract_from_pdf(path: Path) -> str:
    """
    Imports PDF file with PDFReader and concatenates the text.
    """
    try:
        reader = PdfReader(str(path))
        # Combine text from all pages, replacing None with "".
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        st.warning(f"PDF could not be read ({path.name}): {e}")
        return ""

def extract_paragraphs_with_pymupdf(path: Path) -> list[dict]:
    """
    Extracts paragraphs from a PDF using PyMuPDF (fitz).
    Returns a list of Dicts: {"page": int, "block": int, "text": str}.
    Merges short text blocks (titles, definitions) based on visual spacing.
    """
    paras = []
    MIN_WORDS_TO_STANDALONE = 6     # Short lines/titles get appended
    MAX_VERTICAL_GAP = 12           # px: short vertical spacer suggests higher probability of being related

    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))

            merged = []
            prev = None  # Dict: {page, block, text, y1} holds previous text block

            for b in blocks:
                x0, y0, x1, y1, text, block_no = b[:6]
                text = (text or "").strip()
                if not text:
                    continue # Skip empty text

                # Skip non-text elements if present
                if len(b) > 6 and b[6] != 0:
                    continue

                # Initialize first text block
                if prev is None:
                    prev = {"page": page_num, "block": int(block_no), "text": text, "y1": y1}
                    continue

                # Merge heuristic: Glossary term + definition, narrow spacing, very short heading
                small_prev = len(prev["text"].split()) < MIN_WORDS_TO_STANDALONE
                looks_like_term = prev["text"].endswith(":") or prev["text"].istitle()
                small_gap = (y0 - prev["y1"]) <= MAX_VERTICAL_GAP

                if small_prev or looks_like_term or small_gap:
                    # Merge consecutive blocks
                    prev["text"] = (prev["text"].rstrip(":") + ": " + text).strip()
                    prev["y1"] = y1
                else:
                    # Save finished block and start new one
                    merged.append({k: prev[k] for k in ("page", "block", "text")})
                    prev = {"page": page_num, "block": int(block_no), "text": text, "y1": y1}

            # Append last block from page
            if prev:
                merged.append({k: prev[k] for k in ("page", "block", "text")})

            paras.extend(merged)

    return paras


def extract_from_pdf_paragraphs(path: Path) -> str:
    """
    Extracts text paragraphs from a PDF and joins them into a single text string.
    Each paragraph is prefixed with a marker indicating page and block number.
    """
    try:
        paras = extract_paragraphs_with_pymupdf(path)
        if not paras:
            return ""
        return "\n\n".join(f"--- <P:{p['page']}.{p['block']}> ---\n{p['text']}" for p in paras)
    except Exception as e:
        st.warning(f"PDF could not be read ({path.name}): {e}")
        return ""

def extract_from_txt(path: Path) -> str:
    """
    Reads text files with fallback.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()

def load_documents():
    """
    Loads all documents (PDF or TXT) from RAW_DIR and returns list of tuples (filename, text).
    """
    docs = []
    for p in RAW_DIR.glob("*"):
        if p.suffix.lower() == ".pdf":
            txt = extract_from_pdf_paragraphs(p)
        elif p.suffix.lower() in [".txt", ".md"]:
            txt = extract_from_txt(p)
        else:
            continue
        if txt.strip():
            docs.append((p.name, txt))
    return docs


# ---- Chunk-size analysis & evaluation ----

def analyze_text_lengths():
    """
    Calculates basic word count statistics across all loaded documents.
    """
    docs = load_documents()
    if not docs:
        st.warning("No documents found in data/raw")
        return None
    lengths = [len(txt.split()) for _, txt in docs]

    # Compute descriptive statistics for text lengths
    stats = {
        "count": len(lengths),
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "min": np.min(lengths),
        "max": np.max(lengths),
    }
    return stats

def evaluate_chunk_sizes_on_data(sizes=[30, 40, 50,60,80,100,120,160,200,240], sample_frac=0.3):
    """
    Tests different chunk sizes to find the one that produces the lowest inter-chunk similarity.
    """
    model = SentenceTransformer(EMB_MODEL_NAME)
    docs = load_documents()
    if not docs:
        raise RuntimeError("No documents in data/raw")

    results = {}
    for size in sizes:
        all_chunks = []
        for _, text in docs:
            # Split each document into chunks of the given size
            all_chunks.extend(chunk_text(text, size=size))
        if not all_chunks:
            continue

        # Randomly sample a fraction of chunks to save computation time
        sample_n = max(1, int(len(all_chunks) * sample_frac))
        sampled = np.random.choice(all_chunks, sample_n, replace=False)

        # Compute embeddings and cosine similarities
        embs = model.encode(sampled, convert_to_numpy=True, normalize_embeddings=True)
        sims = np.dot(embs, embs.T)

        # Average off-diagonal similarity value (how similar chunks are to each other)
        avg_sim = (np.sum(sims) - len(sims)) / (len(sims)**2 - len(sims))
        results[size] = avg_sim
    return results

def auto_select_chunk_size():
    """
    Computes and displays the optimal chunk size based on inter-chunk similarity.
    """
    stats = analyze_text_lengths()
    if stats:
        st.info(f"📊 {stats['count']} Docs — Average length {stats['mean']:.0f} words "
                f"(Median {stats['median']:.0f})")

    st.write("🔎 Evaluating candidate chunk sizes...")
    results = evaluate_chunk_sizes_on_data()

    # Display evaluation results
    for s, sim in results.items():
        st.write(f"Chunk size {s:3d}: average inter-chunk similarity = {sim:.4f}")

    # Select chunk size with the lowest average similarity (best separation)
    best = min(results, key=results.get)
    st.success(f"Suggested optimal chunk size: {best} words")
    return best


# ---- Index building & retrieval ----

def build_index(chunk_size=200, overlap=20):
    """
    Builds the FAISS index:
        1. Loads and splits documents into chunks.
        2. Computes embeddings using SentenceTransformer.
        3. Stores them in a FAISS index file and JSON metadata.
    """
    model = SentenceTransformer(EMB_MODEL_NAME)
    texts, metas = [], []

    for name, full in load_documents():
        # Split text by paragraph markers so chunks don't cross logical paragraph boundaries
        paragraphs = [p for p in full.split("\n\n") if p.strip()]

        current_page, current_block = None, None
        for para in paragraphs:
            # Detect marker (first line)
            first_line, *rest = para.splitlines()
            m = P_MARK.match(first_line.strip())
            if m:
                current_page, current_block = int(m.group(1)), int(m.group(2))
                para_text = "\n".join(rest).strip()
            else:
                para_text = para.strip()

            # Split paragraph into word chunks with overlap with min size buffer
            buffer_words = []
            for ch in chunk_text(para_text, size=chunk_size, overlap=overlap):
                w = ch.split()
                # If chunk is too short, accumulate words until minimum threshold is reached
                if len(buffer_words) + len(w) < MIN_CHUNK_WORDS:
                    buffer_words.extend(w)
                    continue

                # Enough words: merge buffered ones and finalize chunk
                if buffer_words:
                    w = buffer_words + w
                    buffer_words = []
                merged = " ".join(w)

                # Assign unique chunk-ID
                chunk_id = len(texts)

                texts.append(merged)
                metas.append({
                    "doc": name,
                    "chunk": chunk_id,
                    "page": current_page,
                    "block": current_block,
                })

            # Handle leftover words at the end of a paragraph
            if buffer_words:
                merged = " ".join(buffer_words)
                chunk_id = len(texts)
                texts.append(merged)
                metas.append({
                    "doc": name,
                    "chunk": chunk_id,
                    "page": current_page,
                    "block": current_block,
                })
                buffer_words = []

    if not texts:
        raise RuntimeError("No documents found. Place PDFs or .txt in data/raw.")
    # Compute embeddings
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    # Create FAISS index
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, str(INDEX_PATH))
    with open(TEXTS_PATH, "w", encoding="utf-8") as f:
        for t, m in zip(texts, metas):
            f.write(json.dumps({"text": t, "meta": m}, ensure_ascii=False) + "\n")
    return len(texts)

@st.cache_resource(show_spinner=False)
def load_retriever():
    """
    Loads FAISS index and corresponding text metadata, using Streamlit caching to avoid reloading on every query.
    """
    if not INDEX_PATH.exists() or not TEXTS_PATH.exists():
        return None, None, None
    model = SentenceTransformer(EMB_MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))

    # Read stored text chunks and metadata
    texts, metas = [], []
    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"]); metas.append(obj["meta"])
    return model, index, (texts, metas)

def search(query: str, k=5):
    """
    Retrieves the top-k most semantically similar text chunks for a given query.
    Returns the results along with timing and embeddings for similarity logging.
    """
    model, index, store = load_retriever()
    if model is None:
        st.warning("No index found. Please rebuild index first.")
        return [], 0.0, None  # hits, retrieval_time, hit_embs

    texts, metas = store

    # Encode query and perform vector similarity search
    t0 = time.time()
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idx = index.search(q, k)
    retrieval_time = time.time() - t0

    # Collect top hits with scores and metadata
    hits, hit_texts = [], []
    for s, i in zip(scores[0], idx[0]):
        hits.append({"text": texts[i], "score": float(s), "meta": metas[i]})
        hit_texts.append(texts[i])

    # Encode hit texts for later metric calculation (inter-chunk similarity)
    hit_embs = None
    if hit_texts:
        hit_embs = model.encode(hit_texts, convert_to_numpy=True, normalize_embeddings=True)

    return hits, retrieval_time, hit_embs


def summarize_with_llm(question: str, hits: list) -> str:
    """
    Generates a natural-language answer to a user question using the top retrieved chunks as context.
    Uses a local FLAN-T5 model by default.
    """
    if not USE_LOCAL_LLM:
        snippet = hits[0]["text"][:400] + ("..." if len(hits[0]["text"]) > 400 else "")
        return f"Top relevant context (extract):\n{snippet}"
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        model_id = "google/flan-t5-base"

        # Load lightweight open-source LLM for summarization
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        nlp = pipeline("text2text-generation", model=mdl, tokenizer=tok)

        # Concatenate top-3 retrieved chunks as context (trim to avoid token overflow)
        context = "\n\n".join(h["text"] for h in hits[:3])
        context = context[:4000]  # avoid token limit

        # Prompt instructs model to answer based only on context (RAG principle)
        prompt = f"Answer the question using ONLY the context.\nQuestion: {question}\nContext:\n{context}\nAnswer:"
        out = nlp(prompt, max_new_tokens=180)[0]["generated_text"]
        return out
    except Exception as e:
        return f"(LLM could not be loaded: {e})"


# ---- UI ----

# Configure appearance of Streamlit page
st.set_page_config(page_title="Study Buddy (RAG)", page_icon="📚", layout="wide")
st.title("📚 Study Buddy — Simple RAG over your study docs")

# Create two layout columns
colA, colB = st.columns([3,2])

# Right column: Document indexing and chunk size
with colB:
    st.subheader("Index")
    # Button computing the best chunk size
    if st.button("🤖 Auto-select optimal chunk size"):
        best_size = auto_select_chunk_size()
        st.session_state["best_chunk_size"] = best_size

    # Button to build or rebuild index
    if st.button("🔁 Rebuild index"):
        size = st.session_state.get("best_chunk_size", 200)

        #Saves the optimal chunk size when clicking on button "Auto-select optimal chunk size
        with st.spinner(f"Build index (chunk size {size})…"):
            try:
                n = build_index(chunk_size=size)
                load_retriever.clear()
                st.success(f"Done. {n} Chunks indexed (size={size}).")
            except Exception as e:
                st.error(str(e))
    st.caption("📁 Save your PDFs or .txt files under `data/raw` and then click above.’")

# Left column: Query input and answers
with colA:
    st.subheader("Ask your question")
    # Input Box
    q = st.text_input("Your Question", "")
    # Slider to select number of retrieved chunks shown in results.
    k = st.slider("How many text snippets to display (Top-k) (Top-k)", 3, 8, 5)

    if q:
        from kpi_logger import log_metrics, compute_inter_chunk_similarity

        # Start total timer
        t_start = time.time()

        # Retrieve results + embeddings
        hits, retrieval_time, hit_embs = search(q, k=k)
        if not hits:
            st.stop()

        # Display answer
        with st.expander("💡 Answer", expanded=True):
            ans = summarize_with_llm(q, hits)
            st.write(ans)

        # Stop total timer
        response_latency = time.time() - t_start

        # Compute average inter-chunk similarity (redundancy)
        inter_chunk_sim = compute_inter_chunk_similarity(hit_embs)

        # Log metrics to CSV
        log_metrics(q, response_latency, retrieval_time, inter_chunk_sim)

        # Display retrieved text snippets from source for transparency.
        st.markdown("---")
        st.subheader("Sources")
        for h in hits:
            meta = h["meta"]
            doc = meta.get("doc")
            ch = meta.get("chunk")
            pg = meta.get("page")
            blk = meta.get("block")
            sc = h["score"]

            # Label includes document name, chunk ID, and match score
            label = f"{doc} — chunk {ch} (score {sc:.3f})"

            # Add page and block info if available
            if pg is not None and blk is not None:
                label = f"{label} — page {pg}, block {blk}"

            # Expandable panel shows the actual retrieved text content
            with st.expander(label, expanded=False):
                st.write(h["text"])

