# 📚 RAG-Based PDF/Document Chatbot

👉 **Visit the deployed app here:** [Your Streamlit URL]  

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** using **sentence-aware embeddings** and a **local vector database** to enable question-answering over custom documents such as PDFs.  

---

## 🚀 Project Architecture & Flow

1. **Document Upload & Preprocessing**
   - User uploads PDFs (or text files).
   - Files are converted to raw text using `PyPDF2` / file readers.
   - Text is split into **sentence-aware chunks** (ensuring context preservation).

2. **Embeddings & Vector Store**
   - Each chunk is embedded using **all-MiniLM-L6-v2 (Sentence Transformers)**.
   - Chunks + embeddings are stored in **ChromaDB (local vector store)**.
   - FAISS backend can also be used for scalability.

3. **RAG Pipeline**
   - When a query is asked:
     - Retrieve top-k relevant chunks using vector similarity search.
     - Feed retrieved chunks + query into the LLM.
     - LLM generates a streaming answer with **citations**.

4. **User Interaction**
   - Streamlit interface allows:
     - Uploading documents
     - Asking queries in natural language
     - Viewing streaming answers
     - Seeing **source passages & citation references**
   - Sidebar provides:
     - Vector store stats (# of docs, embeddings)
     - Reset buttons (clear chat history, clear database)

