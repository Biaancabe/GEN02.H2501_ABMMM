# 🎓 Study Buddy (RAG-Based Learning Assistant)

Hi there! My name is Study Buddy and I am your brandnew personal learing assisant, available 24/7!
I’m here to help you study smarter, not harder — by reading, indexing, and understanding your lecture notes for you (because let’s be honest, nobody likes scrolling through 200 slides at 2 AM).

## 💡 What Is Study Budy?

Study Buddy is your personal AI learning partner.  
Upload your lecture slides, PDFs, or notes, then get started with questions. Study Buddy digs through your material, pulls out the useful bits, and crafts a smart, context-aware answer.

## ✨ Features

- 📚 **Document Uploads:** PDFs, text files, lecture notes — I take it all.  
- 🔎 **Smart Search:** Uses semantic similarity (fancy word for “it gets what you mean”).  
- 💬 **Chat Interface:** Ask questions, get answers that actually make sense.  
- 🧠 **RAG-powered Intelligence:** Combines document retrieval with generative AI.  
- 🧩 **Extensible:** Add summaries, flashcards, or quiz modules later.

##  🏗️  Setup Instructions

### 1) Clone the repository
```bash
git clone https://github.com/yourusername/study-buddy.git
cd study-buddy
```

### 2) Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```
## 🚀 Get Started

- 📁 **Place your study files** in the folder: `data/raw`  

- 💻 **Open your terminal** then type the following command to launch the app using the path of the script  
    ```bash
    streamlit run "C:\Users\...\RAG.py"
    ```
## 🧱 Project Structure

```plaintext
study-buddy/
├── app.py              # Main Streamlit application
├── kpi_logger.py       # Logging helper for performance metrics
├── data/
│   ├── raw/            # Place your PDFs or TXT files here
│   └── index/          # Generated embeddings and FAISS index
└── requirements.txt    # Python dependencies