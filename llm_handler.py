from groq import Groq


def initialize_groq_client(api_key):
    """
    Groq client initialize karta hai
    api_key: user ka Groq API key
    """
    try:
        client = Groq(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"Groq client error: {str(e)}"


def summarize_context(client, retrieved_chunks, chat_history, user_question):
    """
    LLM 1 - Mixtral 8x7B
    Retrieved chunks ko summarize karta hai
    """
    try:
        # Chunks ko ek string mein jodo
        chunks_text = "\n\n".join([
            f"Chunk {i+1}:\n{chunk['chunk']}"
            for i, chunk in enumerate(retrieved_chunks)
        ])

        # Chat history ko string mein convert karo
        history_text = ""
        if chat_history:
            history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in chat_history[-4:]  # Last 4 messages lo
            ])

        # Prompt banao
        prompt = f"""You are a helpful assistant that summarizes document context.

Previous Conversation:
{history_text if history_text else "No previous conversation"}

Retrieved Document Chunks:
{chunks_text}

Current Question: {user_question}

Please provide a concise summary of the retrieved chunks that is most relevant 
to answer the current question. Focus only on information present in the chunks."""

        # LLM 1 - Mixtral 8x7B call karo
        response = client.chat.completions.create(
           model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at summarizing document context concisely and accurately."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=500
        )

        summary = response.choices[0].message.content
        return summary, None

    except Exception as e:
        return None, f"Summarization error: {str(e)}"


def generate_answer(client, context_summary, chat_history, user_question):
    """
    LLM 2 - Llama 3.1 8B Instant
    Final answer generate karta hai
    """
    try:
        # Chat history format karo
        messages = [
            {
                "role": "system",
                "content": """You are a helpful PDF assistant. 
Answer questions based on the provided document context.
If the answer is not in the context, say so clearly.
Be accurate, helpful and concise."""
            }
        ]

        # Previous chat history add karo
        if chat_history:
            for msg in chat_history[-6:]:  # Last 6 messages
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Current question with context add karo
        current_prompt = f"""Context Summary from Document:
{context_summary}

User Question: {user_question}

Please answer the question based on the context provided above."""

        messages.append({
            "role": "user",
            "content": current_prompt
        })

        # LLM 2 - Llama 3.1 8B call karo
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.5,
            max_tokens=1000
        )

        answer = response.choices[0].message.content
        return answer, None

    except Exception as e:
        return None, f"Answer generation error: {str(e)}"


def get_response(client, retrieved_chunks, chat_history, user_question):
    """
    Complete Dual LLM Pipeline:
    1. LLM 1 - Context summarize karo
    2. LLM 2 - Answer generate karo
    """
    # Step 1: LLM 1 se context summarize karo
    context_summary, error = summarize_context(
        client,
        retrieved_chunks,
        chat_history,
        user_question
    )
    if error:
        return None, None, error

    # Step 2: LLM 2 se answer generate karo
    answer, error = generate_answer(
        client,
        context_summary,
        chat_history,
        user_question
    )
    if error:
        return None, None, error

    return context_summary, answer, None