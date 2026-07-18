import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_file):
    """
    PDF file se text extract karta hai
    pdf_file: uploaded file object (Streamlit se)
    """
    try:
        # PDF bytes mein convert karo
        pdf_bytes = pdf_file.read()
        
        # PyMuPDF se open karo
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        full_text = ""
        total_pages = len(doc)
        
        # Har page se text nikalo
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()
            full_text += f"\n--- Page {page_num + 1} ---\n"
            full_text += text
        
        doc.close()
        
        if not full_text.strip():
            return None, "PDF mein koi text nahi mila!"
            
        return full_text, None
        
    except Exception as e:
        return None, f"PDF padhne mein error: {str(e)}"


def chunk_text(text):
    """
    Text ko chunks mein todta hai
    1000 characters per chunk
    200 characters overlap
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        
        if not chunks:
            return None, "Text chunks nahi ban sake!"
            
        return chunks, None
        
    except Exception as e:
        return None, f"Chunking mein error: {str(e)}"


def process_pdf(pdf_file):
    """
    Complete PDF processing:
    1. Text extract karo
    2. Chunks banao
    """
    # Step 1: Text extract karo
    text, error = extract_text_from_pdf(pdf_file)
    if error:
        return None, None, error
    
    # Step 2: Chunks banao
    chunks, error = chunk_text(text)
    if error:
        return None, None, error
    
    return text, chunks, None