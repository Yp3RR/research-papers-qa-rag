# 📚 Research Paper Q&A System with RAG

Retrieval-Augmented Generation (RAG) system for question-answering over research papers with source attribution.

## 🎯 Problem

Researchers need to quickly extract specific information from multiple papers. Reading 20+ papers thoroughly is time-consuming. Existing LLM solutions (ChatGPT) have limitations: token limits, high costs, hallucination, and no source verification.

## 💡 Solution

RAG-based Q&A system that:
- Processes multiple PDFs efficiently
- Retrieves only relevant sections
- Generates accurate answers with exact citations
- Runs completely offline (no API costs)

## 🎬 Demo

**Upload PDF → Ask Question → Get Answer + Sources**

Example:
```
Q: "What is BERT's pre-training objective?"
A: "BERT uses masked language modeling and next sentence prediction..."
Sources: Chunk 1 (Page 4), Chunk 3 (Page 5)
```

## 🏗️ Architecture
```
PDF Upload
    ↓
Text Extraction (PyPDF2)
    ↓
Chunking (500 tokens, 50 overlap)
    ↓
Embeddings (sentence-transformers, 384-dim)
    ↓
Vector Storage (FAISS)
    ↓
Query → Similarity Search (top-3 chunks)
    ↓
Context + Question → LLM (Flan-T5)
    ↓
Answer + Source Attribution
```

## 🛠️ Tech Stack

**Core:**
- Python 3.12
- LangChain (RAG orchestration)
- sentence-transformers (embeddings)
- FAISS (vector database)
- Flan-T5 (text generation)

**UI:**
- Streamlit (web interface)

**PDF Processing:**
- PyPDF2

## 📁 Project Structure
```
research-paper-qa-rag/
├── data/pdfs/              # Upload research papers
├── notebooks/              # Development notebooks
├── app/
│   └── streamlit_app.py    # Main application
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Yp3RR/research-paper-qa-rag.git
cd research-paper-qa-rag
pip install -r requirements.txt
```

### 2. Run Application
```bash
cd app
streamlit run streamlit_app.py
```

### 3. Use
1. Upload a PDF research paper
2. Wait for processing (~5-10 seconds)
3. Ask questions in natural language
4. View answer + source chunks

## 🔬 Technical Details

**Chunking Strategy:**
- Size: 500 characters
- Overlap: 50 characters
- Method: RecursiveCharacterTextSplitter

**Embeddings:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Similarity: Cosine

**Retrieval:**
- Vector DB: FAISS (local, CPU)
- Top-k: 3 chunks per query

**Generation:**
- Model: `google/flan-t5-base`
- Max length: 200 tokens
- Device: CPU (no GPU needed)

## 📊 Performance

- **Processing:** ~10 seconds for 20-page PDF
- **Query time:** ~2-3 seconds
- **Accuracy:** Answers grounded in actual document content
- **Cost:** $0 (completely offline)

## 🔮 Future Enhancements

- [ ] Multi-document support (compare across papers)
- [ ] Page number extraction for citations
- [ ] Export Q&A history
- [ ] Support for more file formats (Word, HTML)
- [ ] Improved chunking (preserve paragraph boundaries)
- [ ] GPU acceleration for faster processing

## 👤 Author

**Yash Patil**  
B.E. EEE + M.Sc. Mathematics | BITS Pilani

[GitHub](https://github.com/Yp3RR) | [LinkedIn](https://www.linkedin.com/in/yash-patil-27b060312/)

## 📝 License

Educational/Portfolio Project

---

**Built with:** LangChain, FAISS, HuggingFace Transformers, Streamlit
