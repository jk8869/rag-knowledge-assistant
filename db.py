import os
import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

# Constants
DB_FAISS_PATH = "data/faiss_index.bin"
DB_DATA_PATH = "data/chunks.json"

# Global State
faiss_index = None
stored_chunks = []
bm25_index = None

def init_db():
    """Initializes Vector DB (FAISS) and Keyword DB (BM25)."""
    global faiss_index, stored_chunks, bm25_index
    
    dimension = 1536
    
    # 1. Load Data & FAISS
    if os.path.exists(DB_FAISS_PATH) and os.path.exists(DB_DATA_PATH):
        faiss_index = faiss.read_index(DB_FAISS_PATH)
        with open(DB_DATA_PATH, "r", encoding="utf-8") as f:
            stored_chunks = json.load(f)
        print(f"✅ Loaded {len(stored_chunks)} chunks from disk.")
        
        # 2. Rebuild BM25 Index (It lives in memory)
        print("🔄 Building BM25 keyword index...")
        tokenized_corpus = [chunk.lower().split() for chunk in stored_chunks]
        bm25_index = BM25Okapi(tokenized_corpus)
        print("✅ BM25 Index ready.")
        
    else:
        faiss_index = faiss.IndexFlatL2(dimension)
        stored_chunks = []
        bm25_index = None
        print("⚠️ No database found. Created new.")

def save_db():
    """Persists state to disk."""
    faiss.write_index(faiss_index, DB_FAISS_PATH)
    with open(DB_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(stored_chunks, f)
    print("✅ Database saved.")

def add_to_db(chunks: list, vectors: list):
    """Adds new chunks to both FAISS and BM25."""
    global bm25_index
    
    # Add to FAISS
    vector_array = np.array(vectors).astype('float32')
    faiss_index.add(vector_array)
    
    # Add to Storage
    stored_chunks.extend(chunks)
    
    # Rebuild BM25 (Simple approach: Rebuild on every upload)
    # In a huge system, we would update incrementally, but this is fine for <10k docs
    tokenized_corpus = [chunk.lower().split() for chunk in stored_chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    save_db()

def search_hybrid(query_text: str, query_vector: list, k=3):
    """
    Performs Hybrid Search:
    1. BM25 (Keyword) Search
    2. FAISS (Vector) Search
    3. Merges results (Deduplicates)
    """
    results = []
    seen_indices = set()
    
    # --- 1. Keyword Search (BM25) ---
    if bm25_index:
        tokenized_query = query_text.lower().split()
        # Get top k keyword matches
        # BM25Okapi.get_top_n returns the text, but we need indices to be precise.
        # So we calculate scores manually to get indices.
        doc_scores = bm25_index.get_scores(tokenized_query)
        # Get top k indices
        top_bm25_indices = np.argsort(doc_scores)[::-1][:k]
        
        print(f"DEBUG BM25: Found indices {top_bm25_indices}")
        
        for idx in top_bm25_indices:
            # Only if score is > 0 (it actually found a keyword)
            if doc_scores[idx] > 0:
                results.append(stored_chunks[idx])
                seen_indices.add(idx)

    # --- 2. Vector Search (FAISS) ---
    query_array = np.array([query_vector]).astype('float32')
    distances, indices = faiss_index.search(query_array, k)
    
    print(f"DEBUG FAISS: Found indices {indices[0]}")
    
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx != -1 and idx < len(stored_chunks):
            if idx not in seen_indices:
                # Add threshold check here if you want (e.g., dist < 1.3)
                results.append(stored_chunks[idx])
                seen_indices.add(idx)
    
    return results

def get_stats():
    return {
        "total_documents": len(stored_chunks),
        "bm25_ready": bm25_index is not None
    }