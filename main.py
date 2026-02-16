from typing import List, Dict
import json
from fastapi.responses import StreamingResponse
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
    # 1. Contextualize
    standalone_question = ai.contextualize_question(request.messages, request.question)
    
    # 2. Embed & Search
    question_vector = ai.get_embedding(standalone_question)
    context_chunks = db.search_hybrid(standalone_question, question_vector, k=3)
    
    if not context_chunks:
        # Return a quick stream saying "No info"
        def no_info_generator():
            yield json.dumps({"type": "token", "content": "I don't have enough info."}) + "\n"
        return StreamingResponse(no_info_generator(), media_type="application/x-ndjson")

    # 3. Prepare the Stream Generator
    context_text = "\n\n---\n\n".join(context_chunks)
    
    def response_generator():
        # A. Send Metadata FIRST
        meta_data = {
            "type": "meta",
            "standalone_question": standalone_question,
            "sources": context_chunks
        }
        yield json.dumps(meta_data) + "\n"
        
        # B. Send Tokens
        ai_generator = ai.get_answer_generator(context_text, request.question)
        for token in ai_generator:
            token_data = {"type": "token", "content": token}
            yield json.dumps(token_data) + "\n"

    # 4. Return the Stream
    return StreamingResponse(response_generator(), media_type="application/x-ndjson")