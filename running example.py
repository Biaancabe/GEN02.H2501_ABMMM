# app.py
import os, json, time
from pathlib import Path
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# -------------------------------
# Settings
# -------------------------------
RAW_DIR = Path("data/raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)
IDX_DIR = Path("data/index"); IDX_DIR.mkdir(parents=True, exist_ok=True)
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXTS_PATH = IDX_DIR / "texts.jsonl"
INDEX_PATH = IDX_DIR / "faiss.index"

# Optional LLM (lokal klein via Transformers) – kann man ausschalten
USE_LOCAL_LLM = False  # auf True stellen, wenn ihr torch+transformers installiert habt

# -------------------------------
# Utils
# -------------------------------
def chunk_text(text: str, size=1000, overlap=150):
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
        st.warning(f"PDF konnte nicht gelesen werden ({path.name}): {e}")
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
        if p.suffix.lower() in [".pdf"]:
            txt = extract_from_pdf(p)
        elif p.suffix.lower() in [".txt", ".md"]:
            txt = extract_from_txt(p)
        else:
            continue
        if txt.strip():
            docs.append((p.name, txt))
    return docs

def build_index():
    model = SentenceTransformer(EMB_MODEL_NAME)
    texts, metas = [], []
    for name, full in load_documents():
        for ci, ch in enumerate(chunk_text(full)):
            texts.append(ch)
            metas.append({"doc": name, "chunk": ci})
    if not texts:
        raise RuntimeError("Keine Dokumente gefunden. Lege PDFs oder .txt in data/raw/ ab.")
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
    texts = np.array(texts, dtype=object)
    return model, index, (texts, metas)

def search(query: str, k=5):
    model, index, store = load_retriever()
    if model is None:
        st.warning("Kein Index gefunden. Bitte erst 'Rebuild index' ausführen.")
        return []
    texts, metas = store
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idx = index.search(q, k)
    hits = []
    for s, i in zip(scores[0], idx[0]):
        hits.append({"text": texts[i].item(), "score": float(s), "meta": metas[i]})
    return hits

def summarize_with_llm(question: str, hits: list) -> str:
    # Optional: kleiner lokaler LLM (funktioniert offline, Qualität basal)
    if not USE_LOCAL_LLM:
        # Fallback: „Extraktive“ Antwort – Top-Kontent + kurzer Satz
        snippet = hits[0]["text"][:400] + ("..." if len(hits[0]["text"]) > 400 else "")
        return f"Top relevant context (extract):\n{snippet}\n\n(You can enable a local LLM in the code to get a natural-language summary.)"
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        model_id = "google/flan-t5-small"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        nlp = pipeline("text2text-generation", model=mdl, tokenizer=tok)
        context = "\n\n".join(h["text"] for h in hits[:3])
        prompt = f"Answer the question using ONLY the context. Cite doc and chunk if possible.\nQuestion: {question}\nContext:\n{context}\nAnswer:"
        out = nlp(prompt, max_new_tokens=180)[0]["generated_text"]
        return out
    except Exception as e:
        return f"(LLM konnte nicht geladen werden: {e})"

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Study Buddy (RAG)", page_icon="📚", layout="wide")
st.title("📚 Study Buddy — Simple RAG over your study docs")

colA, colB = st.columns([3,2])

with colB:
    st.subheader("Index")
    if st.button("🔁 Rebuild index", help="Liest PDFs/Texte aus data/raw/, chunked & indexed"):
        with st.spinner("Baue Index…"):
            try:
                n = build_index()
                # Cache invalidieren
                load_retriever.clear()
                st.success(f"Fertig. {n} Chunks indexiert.")
            except Exception as e:
                st.error(str(e))
    st.caption("📁 Lege deine PDFs oder .txt unter `data/raw/` ab und klicke dann oben.")

with colA:
    st.subheader("Ask your question")
    q = st.text_input("Deine Frage (z. B. 'Explain AR(1) stationarity intuitively')", "")
    k = st.slider("Wie viele Text-Snippets einblenden (Top-k)", 3, 8, 5)

    if q:
        hits = search(q, k=k)
        if not hits:
            st.stop()

        # Antwort
        with st.expander("💡 Answer", expanded=True):
            ans = summarize_with_llm(q, hits)
            st.write(ans)

        # Zitate & Kontexte
        st.markdown("---")
        st.subheader("Sources")
        for h in hits:
            doc = h["meta"]["doc"]; ch = h["meta"]["chunk"]; sc = h["score"]
            with st.expander(f"{doc} — chunk {ch} (score {sc:.3f})", expanded=False):
                st.write(h["text"])
