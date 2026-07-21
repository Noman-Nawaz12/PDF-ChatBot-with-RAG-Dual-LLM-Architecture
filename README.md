# 📄 PDF ChatBot with RAG & Dual LLM Architecture

An intelligent PDF Question Answering system built using **Retrieval-Augmented Generation (RAG)** and a **Dual LLM Architecture**. Upload any PDF document and ask questions in natural language. The chatbot retrieves only the most relevant information from the document and generates accurate, context-aware responses.

---

## 🚀 Features

- 📄 Upload and chat with any PDF document
- 🔍 Retrieval-Augmented Generation (RAG)
- 🤖 Dual LLM Architecture for higher quality responses
- 🧠 Conversation memory for contextual discussions
- ⚡ Fast semantic search using FAISS
- 📚 Intelligent document chunking with LangChain
- 🔤 High-quality sentence embeddings
- 💬 Natural language question answering
- 🎨 Clean Streamlit web interface

---

## 🏗️ System Architecture

```text
                PDF Upload
                     │
                     ▼
          PyMuPDF Text Extraction
                     │
                     ▼
          LangChain Text Chunking
                     │
                     ▼
     Sentence Transformer Embeddings
                     │
                     ▼
          FAISS Vector Database
                     │
         User Question Received
                     │
                     ▼
     Semantic Similarity Retrieval
                     │
                     ▼
      Retrieved Relevant Chunks
                     │
         ┌───────────▼───────────┐
         │  LLM 1 (Summarizer)   │
         │ llama-3.3-70b         │
         └───────────┬───────────┘
                     ▼
     Optimized Context Summary
                     │
         ┌───────────▼───────────┐
         │ LLM 2 (Answer Model)  │
         │ llama-3.1-8b-instant  │
         └───────────┬───────────┘
                     ▼
         Intelligent Final Response
```

---

# 📂 Project Structure

```
PDF-ChatBot-with-RAG-Dual-LLM-Architecture/
│
├── app.py                # Streamlit application
├── pdf_processor.py      # PDF text extraction
├── embeddings.py         # Embedding generation
├── rag_pipeline.py       # Retrieval pipeline
├── llm_handler.py        # Dual LLM interaction
├── requirements.txt      # Dependencies
└── README.md
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| Streamlit | User Interface |
| PyMuPDF | PDF Processing |
| LangChain | Text Splitting & RAG |
| Sentence Transformers | Text Embeddings |
| FAISS | Vector Database |
| Groq API | LLM Inference |

---

# 🤖 AI Models Used

## LLM 1 – Context Summarizer

**Model**
```
llama-3.3-70b-versatile
```

Purpose:

- Summarizes retrieved document chunks
- Removes redundant information
- Produces optimized context for the second model

---

## LLM 2 – Answer Generator

**Model**
```
llama-3.1-8b-instant
```

Purpose:

- Generates natural language responses
- Uses summarized context
- Maintains conversational memory

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/PDF-ChatBot-with-RAG-Dual-LLM-Architecture.git

cd PDF-ChatBot-with-RAG-Dual-LLM-Architecture
```

---

## 2. Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Groq API Key

Create a **.env** file in the project root.

```env
GROQ_API_KEY=your_api_key_here
```

---

## 5. Run the Application

```bash
streamlit run app.py
```

The application will launch in your browser.

---

# 💡 How It Works

### Step 1

Upload a PDF document.

↓

### Step 2

PyMuPDF extracts the text.

↓

### Step 3

LangChain splits the text into smaller chunks.

↓

### Step 4

Sentence Transformers generate embeddings.

↓

### Step 5

Embeddings are stored inside FAISS.

↓

### Step 6

User asks a question.

↓

### Step 7

FAISS retrieves the most relevant chunks.

↓

### Step 8

LLM 1 summarizes the retrieved context.

↓

### Step 9

LLM 2 generates the final conversational answer.

---

# 📦 Dependencies

Some of the major libraries used include:

- streamlit
- langchain
- sentence-transformers
- faiss-cpu
- pymupdf
- python-dotenv
- groq

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# 🎯 Use Cases

- Research Paper Assistant
- Academic PDF Chat
- Legal Document Analysis
- Company Reports
- Business Documentation
- User Manuals
- Technical Documentation
- E-books
- Study Notes

---

# 📈 Future Improvements

- Support multiple PDFs
- Citation and source highlighting
- Hybrid Search (Keyword + Semantic)
- PDF OCR support
- Voice input/output
- Multi-language support
- Chat history export
- Cloud deployment

---

# 👨‍💻 Author

**Noman Nawaz**

Software Engineering Student

Interested in:

- Artificial Intelligence
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- NLP
- Machine Learning

---

