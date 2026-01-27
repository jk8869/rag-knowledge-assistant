from fastapi import FastAPI
from fastapi import UploadFile, File
from utils import extract_text_from_pdf

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "running", "service": "knowledge-assistant"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # We need to read the file into a stream for pypdf
    text_content = extract_text_from_pdf(file.file)

    # For now, just return the first 500 chars to prove it worked
    return {
        "filename": file.filename,
        "preview": text_content[:500]
    }

