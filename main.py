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
    # 1. THE ROUTER: Decide what to do
    intent = ai.classify_intent(request.messages, request.question)
    
    # --- BRANCH A: GENERAL CHAT (Fast Path) ---
    if intent == "chat":
        def chat_generator():
            # Send metadata (No sources for chat)
            meta_data = {
                "type": "meta",
                "intent": "chat", # Let frontend know mode
                "sources": [] 
            }
            yield json.dumps(meta_data) + "\n"
            
            # Stream the chat response
            chat_stream = ai.get_general_chat_generator(request.messages, request.question)
            for token in chat_stream:
                token_data = {"type": "token", "content": token}
                yield json.dumps(token_data) + "\n"

        return StreamingResponse(chat_generator(), media_type="application/x-ndjson")

    # --- BRANCH B: KNOWLEDGE SEARCH (Deep Path) ---
    elif intent == "search":
        # 2. Contextualize (Rewrite)
        standalone_question = ai.contextualize_question(request.messages, request.question)
        
        # 3. Embed & Search
        question_vector = ai.get_embedding(standalone_question)
        context_chunks = db.search_hybrid(standalone_question, question_vector, k=3)
        
        if not context_chunks:
            def no_info_generator():
                yield json.dumps({"type": "token", "content": "I couldn't find that in the documents."}) + "\n"
            return StreamingResponse(no_info_generator(), media_type="application/x-ndjson")

        # 4. Stream Answer
        context_text = "\n\n---\n\n".join(context_chunks)
        
        def rag_generator():
            # Send Metadata (With Sources)
            meta_data = {
                "type": "meta",
                "intent": "search",
                "standalone_question": standalone_question,
                "sources": context_chunks
            }
            yield json.dumps(meta_data) + "\n"
            
            # Stream RAG response
            rag_stream = ai.get_answer_generator(context_text, request.question)
            for token in rag_stream:
                token_data = {"type": "token", "content": token}
                yield json.dumps(token_data) + "\n"

        return StreamingResponse(rag_generator(), media_type="application/x-ndjson")