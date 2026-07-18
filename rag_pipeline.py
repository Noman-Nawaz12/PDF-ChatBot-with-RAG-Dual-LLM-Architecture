from embeddings import search_similar_chunks
from llm_handler import get_response


def run_rag_pipeline(user_question, index, chunks, client, chat_history, top_k=3):
    """
    Complete RAG Pipeline:
    1. Similar chunks dhundo
    2. Dual LLM se answer lo
    
    user_question: user ka sawaal
    index: FAISS index
    chunks: text chunks list
    client: Groq client
    chat_history: purani baatein
    top_k: kitne chunks chahiye
    """
    try:
        # Step 1: Similar chunks dhundo
        retrieved_chunks, error = search_similar_chunks(
            query=user_question,
            index=index,
            chunks=chunks,
            top_k=top_k
        )
        
        if error:
            return None, None, None, f"Chunks dhundne mein error: {error}"
        
        if not retrieved_chunks:
            return None, None, None, "Koi relevant chunks nahi mile!"

        # Step 2: Dual LLM se response lo
        context_summary, answer, error = get_response(
            client=client,
            retrieved_chunks=retrieved_chunks,
            chat_history=chat_history,
            user_question=user_question
        )
        
        if error:
            return None, None, None, f"Response generate karne mein error: {error}"

        return retrieved_chunks, context_summary, answer, None

    except Exception as e:
        return None, None, None, f"RAG pipeline error: {str(e)}"


def format_retrieved_chunks(retrieved_chunks):
    """
    Retrieved chunks ko display ke liye format karta hai
    """
    formatted = []
    for i, chunk in enumerate(retrieved_chunks):
        formatted.append(
            f"📄 Chunk {i+1} "
            f"(Similarity Score: {chunk['distance']:.4f}):\n"
            f"{chunk['chunk']}\n"
            f"{'-'*50}"
        )
    return "\n\n".join(formatted)