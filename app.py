import streamlit as st
from dotenv import load_dotenv
import os
from pdf_processor import process_pdf
from embeddings import process_embeddings
from rag_pipeline import run_rag_pipeline, format_retrieved_chunks
from llm_handler import initialize_groq_client

# Environment variables load karo
load_dotenv()

# ─────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────
st.set_page_config(
    page_title="PDF ChatBot",
    page_icon="📄",
    layout="wide"
)

# ─────────────────────────────────────────
# Session State Initialize karo
# ─────────────────────────────────────────
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "index" not in st.session_state:
        st.session_state.index = None
    if "client" not in st.session_state:
        st.session_state.client = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None

initialize_session_state()

# ─────────────────────────────────────────
# Main Title
# ─────────────────────────────────────────
st.title("📄 PDF ChatBot")
st.markdown("##### RAG + Dual LLM Architecture (Groq)")
st.divider()

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    st.subheader("🔑 Groq API Key")
    api_key = st.text_input(
        "Enter your Groq API Key",
        type="password",
        placeholder="gsk_...",
        value=os.getenv("GROQ_API_KEY", "")
    )
    
    if api_key:
        client, error = initialize_groq_client(api_key)
        if error:
            st.error(f"❌ API Key Error: {error}")
        else:
            st.session_state.client = client
            st.success("✅ API Key Connected!")
    
    st.divider()
    
    # PDF Upload
    st.subheader("📂 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )
    
    # Retrieval Settings
    st.divider()
    st.subheader("🎯 Retrieval Settings")
    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=5,
        value=3,
        help="Zyada chunks = zyada context, lekin slow"
    )
    
    # Process PDF Button
    st.divider()
    if uploaded_file and api_key:
        if st.button("🚀 Process PDF", use_container_width=True):
            with st.spinner("PDF process ho raha hai..."):
                
                # Step 1: PDF se text aur chunks nikalo
                st.info("📖 PDF se text nikal raha hai...")
                text, chunks, error = process_pdf(uploaded_file)
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.info(f"✅ {len(chunks)} chunks bane!")
                    
                    # Step 2: Embeddings aur FAISS index banao
                    st.info("🔢 Embeddings ban rahi hain...")
                    embeddings, index, error = process_embeddings(chunks)
                    
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        # Session state mein save karo
                        st.session_state.chunks = chunks
                        st.session_state.index = index
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.chat_history = []
                        st.success("✅ PDF ready hai! Ab sawaal pucho!")
    
    # Clear Chat Button
    if st.session_state.pdf_processed:
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # PDF Info
    if st.session_state.pdf_processed:
        st.divider()
        st.success(f"📄 **Active PDF:**\n{st.session_state.pdf_name}")
        st.info(f"📦 **Total Chunks:** {len(st.session_state.chunks)}")

# ─────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────
if not st.session_state.pdf_processed:
    # Welcome Screen
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 1️⃣ API Key\nGroq API key sidebar mein daalo")
    
    with col2:
        st.info("### 2️⃣ PDF Upload\nApna PDF upload karo")
    
    with col3:
        st.info("### 3️⃣ Chat\nSawaal pucho aur jawab pao!")

else:
    # Chat History dikhao
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Context aur chunks expandable mein dikhao
                if "context_summary" in message:
                    with st.expander("📝 Context Summary dekho"):
                        st.write(message["context_summary"])
                
                if "retrieved_chunks" in message:
                    with st.expander("📄 Retrieved Chunks dekho"):
                        st.text(format_retrieved_chunks(
                            message["retrieved_chunks"]
                        ))

    # Chat Input
    user_question = st.chat_input(
        "PDF ke baare mein kuch bhi pucho..."
    )
    
    if user_question:
        # User message dikhao
        with st.chat_message("user"):
            st.write(user_question)
        
        # Chat history mein add karo
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        # RAG Pipeline chalao
        with st.chat_message("assistant"):
            with st.spinner("Soch raha hun... 🤔"):
                
                retrieved_chunks, context_summary, answer, error = run_rag_pipeline(
                    user_question=user_question,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    client=st.session_state.client,
                    chat_history=st.session_state.chat_history,
                    top_k=top_k
                )
                
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    # Answer dikhao
                    st.write(answer)
                    
                    # Expandable sections
                    with st.expander("📝 Context Summary dekho"):
                        st.write(context_summary)
                    
                    with st.expander("📄 Retrieved Chunks dekho"):
                        st.text(format_retrieved_chunks(retrieved_chunks))
                    
                    # Chat history mein save karo
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "context_summary": context_summary,
                        "retrieved_chunks": retrieved_chunks
                    })