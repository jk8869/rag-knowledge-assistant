from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import our new modules
from utils import extract_text_from_pdf, chunk_text
import ai
import db

# Lifecycle: Start DB on boot
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield

app = FastAPI(title="Internal Knowledge Assistant", lifespan=lifespan)

# --- Routes ---

@app.get("/health")
def health_check():
    return {"status": "healthy", "stats": db.get_stats()}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. Process File
    raw_text = extract_text_from_pdf(file.file)
    chunks = chunk_text(raw_text)
    
    # 2. Generate Embeddings
    vectors = [ai.get_embedding(chunk) for chunk in chunks]
    
    # 3. Store in DB
    db.add_to_db(chunks, vectors)
    
    return {"filename": file.filename, "chunks_added": len(chunks)}

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    # 1. Embed Question (for FAISS)
    question_vector = ai.get_embedding(request.question)
    
    # 2. Retrieve Context (Hybrid: Uses both text AND vector)
    # We pass 'request.question' for BM25 and 'question_vector' for FAISS
    context_chunks = db.search_hybrid(request.question, question_vector, k=3)
    
    if not context_chunks:
        return {"answer": "I don't have enough info."}
    
    # 3. Generate Answer
    context_text = "\n\n---\n\n".join(context_chunks)
    answer = ai.get_answer(context_text, request.question)
    
    return {
        "question": request.question,
        "answer": answer,
        "sources": context_chunks
    }