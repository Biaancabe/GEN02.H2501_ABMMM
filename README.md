# 🎓 Study Buddy (RAG-Based Learning Assistant)

Hi there! My name is Study Buddy and I am your brandnew personal learing assisant, available 24/7!
I’m here to help you study smarter, not harder — by reading, indexing, and understanding your lecture notes for you (because let’s be honest, nobody likes scrolling through 200 slides at 2 AM).

## 💡 What Is Study Budy?

Study Buddy is your personal AI learning partner.  
Upload your lecture slides, PDFs, or notes, then get started with questions. Study Buddy digs through your material, pulls out the useful bits, and crafts a smart, context-aware answer.

## ✨ Features

- 📚 Upload your study materials — PDFs, text files, and lecture notes.
- ✂️ Smart text chunking with automatic size optimization for better retrieval.
- 🧠 Semantic embeddings via all-MiniLM-L6-v2 for meaning-based search.
- ⚡ Fast retrieval powered by FAISS vector similarity search.
- 💬 Local Q&A using Flan-T5, fully offline and privacy-safe.
- 🧮 Streamlit interface for interactive querying and viewing sources.
- 📊 Performance logging for response time and retrieval quality.

##  🏗️  Setup Instructions

### 1) Clone the repository
```bash
git clone https://github.com/Biaancabe/GEN02.H2501_ABMMM.git
cd GEN02.H2501_ABMMM
```

### 2) Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows (Git Bash): venv/Scripts/activate 
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```
## 🚀 Get Started

- 📁 **Place your study files** in the folder: `data/raw`  

- 💻 **Open your terminal** navigate to you folder GEN02.H2501_ABMMM then type the following command to launch the app using the path of the script. Side note: This will take some time. 
    ```bash
    streamlit run RAG.py
    ```
## 🧱 Project Structure

```plaintext
GEN02.H2501_ABMMM/
├── RAG.py              # Main Streamlit application
├── kpi_logger.py       # Logging helper for performance metrics
├── analyze_kpi.py      # Create simple plots for visualizing purposes
├── data/
│   ├── raw/            # Place your PDFs or TXT files here
│   └── index/          # Generated embeddings and FAISS index
└── requirements.txt    # Python dependencies
