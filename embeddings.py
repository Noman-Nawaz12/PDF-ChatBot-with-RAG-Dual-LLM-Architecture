import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def get_model():
    """Model load karta hai"""
    return SentenceTransformer('all-MiniLM-L6-v2')


def create_embeddings(chunks):
    """Text chunks ko vector embeddings mein convert karta hai"""
    try:
        model = get_model()
        embeddings = model.encode(
            chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings, None
    except Exception as e:
        return None, f"Embeddings banane mein error: {str(e)}"


def create_faiss_index(embeddings):
    """FAISS vector index banata hai"""
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))
        return index, None
    except Exception as e:
        return None, f"FAISS index banane mein error: {str(e)}"


def search_similar_chunks(query, index, chunks, top_k=3):
    """Query ke liye similar chunks dhundta hai"""
    try:
        model = get_model()
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True
        )
        distances, indices = index.search(
            query_embedding.astype(np.float32),
            top_k
        )
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                relevant_chunks.append({
                    "chunk": chunks[idx],
                    "distance": float(distances[0][i]),
                    "index": int(idx)
                })
        return relevant_chunks, None
    except Exception as e:
        return None, f"Search mein error: {str(e)}"


def process_embeddings(chunks):
    """Complete embedding process"""
    embeddings, error = create_embeddings(chunks)
    if error:
        return None, None, error
    index, error = create_faiss_index(embeddings)
    if error:
        return None, None, error
    return embeddings, index, None