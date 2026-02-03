# 🤖 Internal Knowledge Assistant (RAG System)

A containerized, AI-powered document retrieval system that allows users to chat with their internal PDF documents. Built with **FastAPI**, **Docker**, **OpenAI**, and **FAISS**.

## 🚀 Key Features
* **RAG Architecture:** Retrieval-Augmented Generation for grounded, accurate AI responses.
* **Vector Search:** Uses FAISS for high-performance similarity search (1536-dim embeddings).
* **Intelligent Filtering:** Implements L2 distance thresholding to reduce hallucinations and token costs.
* **Microservices:** Fully containerized Backend (FastAPI) and Frontend (Streamlit) using Docker Compose.
* **Persistence:** Volume-mapped storage ensures data survives container restarts.

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, Uvicorn
* **AI/ML:** OpenAI API (Embeddings + GPT-4o), FAISS (Vector DB)
* **Frontend:** Streamlit
* **Infrastructure:** Docker, Docker Compose

## 📦 How to Run
1.  **Clone the repo:**
    ```bash
    git clone <your-repo-url>
    ```
2.  **Add API Key:**
    Create a `.env` file and add: `OPENAI_API_KEY=sk-...`
3.  **Launch:**
    ```bash
    docker compose up --build
    ```
4.  **Access:**
    Open `http://localhost:8501` to chat with your documents.