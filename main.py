from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "running", "service": "knowledge-assistant"}
