import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from fastapi import UploadFile, File
from utils import extract_text_from_pdf, chunk_text

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

def get_embedding(text: str):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small" # Efficient and cheap model
    )
    return response.data[0].embedding

@app.get("/health")
def health_check():
    return {"status": "running", "service": "knowledge-assistant"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. Extract Text
    raw_text = extract_text_from_pdf(file.file)

    # 2. Chunk Text
    chunks = chunk_text(raw_text)

    # 3. Embed the first chunk (Just to test connection)
    first_chunk_vector = get_embedding(chunks[0])

    return {
        "filename": file.filename,
        "total_chunks": len(chunks),
        "first_chunk_preview": chunks[0][:100],
        "embedding_length": len(first_chunk_vector), # Should be 1536 for OpenAI
        "sample_vector_data": first_chunk_vector[:5] # Show first 5 numbers
    }

