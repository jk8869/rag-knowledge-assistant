import os
import json
import faiss
import numpy as np

DB_FAISS_PATH = "faiss_index.bin"
DB_DATA_PATH = "chunks.json"

# Global State
faiss_index = None
stored_chunks = []

def init_db():
    """Initializes the vector DB (loads from disk or creates new)."""
    global faiss_index, stored_chunks
    
    dimension = 1536
    
    if os.path.exists(DB_FAISS_PATH) and os.path.exists(DB_DATA_PATH):
        faiss_index = faiss.read_index(DB_FAISS_PATH)
        with open(DB_DATA_PATH, "r", encoding="utf-8") as f:
            stored_chunks = json.load(f)
        print("✅ Database loaded from disk.")
    else:
        faiss_index = faiss.IndexFlatL2(dimension)
        stored_chunks = []
        print("⚠️ No database found. Created new.")

def save_db():
    """Persists state to disk."""
    faiss.write_index(faiss_index, DB_FAISS_PATH)
    with open(DB_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(stored_chunks, f)
    print("✅ Database saved.")

def add_to_db(chunks: list, vectors: list):
    """Adds new chunks and vectors to memory."""
    # Add vectors to FAISS
    vector_array = np.array(vectors).astype('float32')
    faiss_index.add(vector_array)
    
    # Add text to list
    stored_chunks.extend(chunks)
    
    # Auto-save
    save_db()

def search_db(query_vector, k=3, threshold=0.5): # Added threshold
    query_array = np.array([query_vector]).astype('float32')
    distances, indices = faiss_index.search(query_array, k)
    
    results = []
    # Loop through both indices AND distances
    for i in range(len(indices[0])):
        idx = indices[0][i]
        dist = distances[0][i]
        
        # 1. Check if it's a valid index
        if idx == -1 or idx >= len(stored_chunks):
            continue
            
        # 2. THE SMART FILTER: Check if it's "close enough"
        # Note: In L2 distance, LOWER is BETTER (0 = exact match)
        if dist < threshold:
            results.append(stored_chunks[idx])
            
    return results

def get_stats():
    return {
        "total_documents": faiss_index.ntotal if faiss_index else 0
    }