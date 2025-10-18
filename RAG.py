from pathlib import Path
import os, json, time
from pathlib import Path
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# ---- Directories ----

RAW_DIR = Path("data/raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)
IDX_DIR = Path("data/index"); IDX_DIR.mkdir(parents=True, exist_ok=True)
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXTS_PATH = IDX_DIR / "texts.jsonl"
INDEX_PATH = IDX_DIR / "faiss.index"
USE_LOCAL_LLM = True


# ---- File loading ----

def chunk_text(text: str, size=200, overlap=20):
    """Split text into overlapping chunks of roughly `size` words."""
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+size]))
        i += max(1, size - overlap)
    return out

def extract_from_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        st.warning(f"PDF could not be read ({path.name}): {e}")
        return ""

def extract_from_txt(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()

def load_documents():
    docs = []
    for p in RAW_DIR.glob("*"):
        if p.suffix.lower() == ".pdf":
            txt = extract_from_pdf(p)
        elif p.suffix.lower() in [".txt", ".md"]:
            txt = extract_from_txt(p)
        else:
            continue
        if txt.strip():
            docs.append((p.name, txt))
    return docs


# ---- Chunk-size analysis & evaluation ----

def analyze_text_lengths():
    docs = load_documents()
    if not docs:
        st.warning("No documents found in data/raw")
        return None
    lengths = [len(txt.split()) for _, txt in docs]
    stats = {
        "count": len(lengths),
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "min": np.min(lengths),
        "max": np.max(lengths),
    }
    return stats

def evaluate_chunk_sizes_on_data(sizes=[100, 200, 300, 400], sample_frac=0.3):
    model = SentenceTransformer(EMB_MODEL_NAME)
    docs = load_documents()
    if not docs:
        raise RuntimeError("No documents in data/raw")

    results = {}
    for size in sizes:
        all_chunks = []
        for _, text in docs:
            all_chunks.extend(chunk_text(text, size=size))
        if not all_chunks:
            continue
        sample_n = max(1, int(len(all_chunks) * sample_frac))
        sampled = np.random.choice(all_chunks, sample_n, replace=False)
        embs = model.encode(sampled, convert_to_numpy=True, normalize_embeddings=True)
        sims = np.dot(embs, embs.T)
        avg_sim = (np.sum(sims) - len(sims)) / (len(sims)**2 - len(sims))
        results[size] = avg_sim
    return results

def auto_select_chunk_size():
    stats = analyze_text_lengths()
    if stats:
        st.info(f"📊 {stats['count']} Docs — Average length {stats['mean']:.0f} words "
                f"(Median {stats['median']:.0f})")

    st.write("🔎 Evaluating candidate chunk sizes...")
    results = evaluate_chunk_sizes_on_data()
    for s, sim in results.items():
        st.write(f"Chunk size {s:3d}: average inter-chunk similarity = {sim:.4f}")
    best = min(results, key=results.get)
    st.success(f"Suggested optimal chunk size: {best} words")
    return best


# ---- Index building & retrieval ----

def build_index(chunk_size=200, overlap=20):
    model = SentenceTransformer(EMB_MODEL_NAME)
    texts, metas = [], []
    for name, full in load_documents():
        for ci, ch in enumerate(chunk_text(full, size=chunk_size, overlap=overlap)):
            texts.append(ch)
            metas.append({"doc": name, "chunk": ci})
    if not texts:
        raise RuntimeError("No documents found. Place PDFs or .txt in data/raw.")
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
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
    if not INDEX_PATH.exists() or not TEXTS_PATH.exists():
        return None, None, None
    model = SentenceTransformer(EMB_MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))
    texts, metas = [], []
    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"]); metas.append(obj["meta"])
    return model, index, (texts, metas)

def search(query: str, k=5):
    model, index, store = load_retriever()
    if model is None:
        st.warning("No index found. Please rebuild index first.")
        return []
    texts, metas = store
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idx = index.search(q, k)
    hits = []
    for s, i in zip(scores[0], idx[0]):
        hits.append({"text": texts[i], "score": float(s), "meta": metas[i]})
    return hits

def summarize_with_llm(question: str, hits: list) -> str:
    if not USE_LOCAL_LLM:
        snippet = hits[0]["text"][:400] + ("..." if len(hits[0]["text"]) > 400 else "")
        return f"Top relevant context (extract):\n{snippet}"
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        model_id = "google/flan-t5-small"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        nlp = pipeline("text2text-generation", model=mdl, tokenizer=tok)
        context = "\n\n".join(h["text"] for h in hits[:3])
        context = context[:4000]  # avoid token limit
        prompt = f"Answer the question using ONLY the context.\nQuestion: {question}\nContext:\n{context}\nAnswer:"
        out = nlp(prompt, max_new_tokens=180)[0]["generated_text"]
        return out
    except Exception as e:
        return f"(LLM could not be loaded: {e})"


# ---- UI ----

st.set_page_config(page_title="Study Buddy (RAG)", page_icon="📚", layout="wide")
st.title("📚 Study Buddy — Simple RAG over your study docs")

colA, colB = st.columns([3,2])

with colB:
    st.subheader("Index")
    if st.button("🤖 Auto-select optimal chunk size"):
        best_size = auto_select_chunk_size()
        st.session_state["best_chunk_size"] = best_size

    if st.button("🔁 Rebuild index"):
        size = st.session_state.get("best_chunk_size", 200)
        with st.spinner(f"Build index (chunk size {size})…"):
            try:
                n = build_index(chunk_size=size)
                load_retriever.clear()
                st.success(f"Done. {n} Chunks indexed (size={size}).")
            except Exception as e:
                st.error(str(e))
    st.caption("📁 Save your PDFs or .txt files under `data/raw` and then click above.’")

with colA:
    st.subheader("Ask your question")
    q = st.text_input("Your Question", "")
    k = st.slider("How many text snippets to display (Top-k) (Top-k)", 3, 8, 5)

    if q:
        hits = search(q, k=k)
        if not hits:
            st.stop()
        with st.expander("💡 Answer", expanded=True):
            ans = summarize_with_llm(q, hits)
            st.write(ans)
        st.markdown("---")
        st.subheader("Sources")
        for h in hits:
            doc = h["meta"]["doc"]; ch = h["meta"]["chunk"]; sc = h["score"]
            with st.expander(f"{doc} — chunk {ch} (score {sc:.3f})", expanded=False):
                st.write(h["text"])
