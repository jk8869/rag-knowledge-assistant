from typing import List, Dict
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
    messages: List[Dict[str, str]] = []

@app.post("/ask")
def ask_question(request: QueryRequest):
    # 1. Contextualize (Rewrite) the Question
    # We use the history to clarify "it", "that", "he", etc.
    standalone_question = ai.contextualize_question(request.messages, request.question)

    # 2. Embed the NEW standalone question
    question_vector = ai.get_embedding(standalone_question)

    # 3. Retrieve Context (using the REWRITTEN question)
    context_chunks = db.search_hybrid(standalone_question, question_vector, k=3)

    if not context_chunks:
        return {"answer": "I don't have enough info based on the documents."}

    # 4. Generate Answer
    # We still send the ORIGINAL question to the final chat model, 
    # but we provide the chunks found by the STANDALONE question.
    context_text = "\n\n---\n\n".join(context_chunks)
    answer = ai.get_answer(context_text, request.question)

    return {
        "question": request.question,
        "standalone_question": standalone_question, # Return this for debugging
        "answer": answer,
        "sources": context_chunks
    }