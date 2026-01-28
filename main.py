import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from fastapi import UploadFile, File
from utils import extract_text_from_pdf, chunk_text
import numpy as np
import faiss
from pydantic import BaseModel # We need this for the search request body
from contextlib import asynccontextmanager

DB_FAISS_PATH = "faiss_index.bin"
DB_DATA_PATH = "chunks.json"

class QueryRequest(BaseModel):
    question: str

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# VECTOR DB SETUP
dimension = 1536 # Must match OpenAI's embedding size
# IndexFlatL2 measures "distance" (L2) between points.
faiss_index = faiss.IndexFlatL2(dimension)
# Simple storage for the text chunks (FAISS only stores numbers, not text)
stored_chunks = []

def get_embedding(text: str):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small" # Efficient and cheap model
    )
    return response.data[0].embedding

def save_db():
    """Saves the FAISS index and the text chunks to disk."""
    # 1. Save FAISS Index (The Math)
    faiss.write_index(faiss_index, DB_FAISS_PATH)
    
    # 2. Save Text Chunks (The Content)
    with open(DB_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(stored_chunks, f)
    print("✅ Database saved to disk.")

def load_db():
    """Loads the FAISS index and text chunks from disk (if they exist)."""
    global faiss_index, stored_chunks
    
    # Check if files exist
    if os.path.exists(DB_FAISS_PATH) and os.path.exists(DB_DATA_PATH):
        # 1. Load FAISS Index
        faiss_index = faiss.read_index(DB_FAISS_PATH)
        
        # 2. Load Text Chunks
        with open(DB_DATA_PATH, "r", encoding="utf-8") as f:
            stored_chunks = json.load(f)
        print("✅ Database loaded from disk.")
    else:
        print("⚠️ No database found on disk. Starting fresh.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the DB
    load_db()
    yield
    # Shutdown: (Optional) We could save here, but saving on upload is safer against crashes
    
app = FastAPI(title="Internal Knowledge Assistant", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "running", "service": "knowledge-assistant"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. Extract & Chunk
    raw_text = extract_text_from_pdf(file.file)
    chunks = chunk_text(raw_text)
    
    # 2. Embed ALL chunks (not just the first one)
    # Note: In production, we'd batch this. For now, a loop is fine for small files.
    vectors = []
    for chunk in chunks:
        vector = get_embedding(chunk)
        vectors.append(vector)
    
    # 3. Add to FAISS (The "Brain")
    # Convert list of lists to numpy array
    vector_array = np.array(vectors).astype('float32')
    faiss_index.add(vector_array)
    
    # 4. Store the text so we can look it up later
    stored_chunks.extend(chunks)

    save_db()
    
    return {
        "status": "success",
        "filename": file.filename,
        "chunks_added": len(chunks),
        "total_documents_in_db": faiss_index.ntotal
    }

@app.post("/search")
def search_knowledge(request: QueryRequest):
    # 1. Embed the user's question
    question_vector = get_embedding(request.question)
    question_array = np.array([question_vector]).astype('float32')
    
    # 2. Search FAISS for the 3 most similar chunks
    k = 3
    distances, indices = faiss_index.search(question_array, k)
    
    # 3. Retrieve the actual text results
    results = []
    for idx in indices[0]:
        if idx < len(stored_chunks):
            results.append(stored_chunks[idx])
            
    return {"results": results}

@app.post("/ask")
def ask_question(request: QueryRequest):
    # 1. Convert question to vector
    question_vector = get_embedding(request.question)
    question_array = np.array([question_vector]).astype('float32')
    
    # 2. Search FAISS for the 3 most relevant chunks
    k = 3
    distances, indices = faiss_index.search(question_array, k)
    
    # 3. Retrieve the text for those chunks
    relevant_chunks = []
    for idx in indices[0]:
        # ADD THIS CHECK: "if idx != -1"
        if idx != -1 and idx < len(stored_chunks):
            relevant_chunks.append(stored_chunks[idx])
            
    # Safety Check: If we found nothing, handle it gracefully
    if not relevant_chunks:
        return {"answer": "I don't have any documents in my memory right now. Please upload a PDF first."}
            
    # 4. Construct the Context (Join the chunks into one big string)
    context_text = "\n\n---\n\n".join(relevant_chunks)
    
    # 5. Send to OpenAI Chat Completion (The "Generation" step)
    system_prompt = """You are a helpful assistant. 
    Answer the user's question strictly based on the provided context. 
    If the answer is not in the context, say 'I don't know'.
    The context might be messy text extracted from a PDF; do your best to clean it up."""

    user_message = f"Context:\n{context_text}\n\nQuestion: {request.question}"

    chat_response = client.chat.completions.create(
        model="gpt-4o-mini", # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    
    answer = chat_response.choices[0].message.content
    
    return {
        "question": request.question,
        "answer": answer,
        "source_chunks": relevant_chunks # Optional: return sources for debugging
    }