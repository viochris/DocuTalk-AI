import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader

def init_state():
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = []
    if "message" not in st.session_state:
        st.session_state["message"] = []

def change_on_api_key():
    st.session_state.pop("agent_executor", None)
    st.session_state.pop("vectorstore", None)
    st.session_state.pop("processed_files", None)
    st.session_state["message"] = []
    st.toast("API Key changed! System reset.", icon="🔄")

def reset_state():
    """Clear chat history to start a fresh conversation session."""
    st.session_state.pop("agent_executor", None)
    st.session_state.pop("vectorstore", None)
    st.session_state.pop("processed_files", None)
    st.session_state["message"] = []
    init_state()
    st.toast("Conversation & Memory Cleared!", icon="🧹")

def reset_agent():
    """Clear chat history to start a fresh conversation session."""
    st.session_state.pop("agent_executor", None)

def process_file(uploaded_file: str):
    """
    Processes an uploaded file (PDF, CSV, TXT, MD) and extracts its content.

    This function creates a temporary file to store the uploaded content,
    selects the appropriate LangChain loader based on the file extension,
    and returns a list of Document objects.

    Args:
        uploaded_file: The file object uploaded via Streamlit.

    Returns:
        List[Document] | None: A list of LangChain Document objects if successful, 
                               or None if the format is unsupported or an error occurs.
    """
    
    # Get the file name
    filename = uploaded_file.name
    # Get the file extension
    file_ext = os.path.splitext(filename)[-1]

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = None

    try:
        # Check file type based on extension (Exact logic as requested)
        if file_ext.lower() == ".pdf" or filename.lower().endswith(".pdf"):
            # Load PDF file
            loader = PyPDFLoader(tmp_file_path)
        elif file_ext.lower() == ".csv" or filename.lower().endswith(".csv"):
            # Load CSV file with UTF-8 encoding
            loader = CSVLoader(tmp_file_path, encoding="utf-8")
        elif file_ext.lower() in [".txt", ".md"] or filename.lower().endswith((".txt", ".md")):
            # Load Text or Markdown file
            loader = TextLoader(tmp_file_path, encoding="utf-8")
        else:
            # Display warning for unsupported file types
            st.warning(f"⚠️ Sorry, the file format '{file_ext}' is not supported. Please upload CSV, PDF, MD, or TXT files only.")
            return None

        return loader.load()

    # --- ADVANCED ERROR HANDLING (ADAPTED FOR FILE ISSUES) ---
    except Exception as e:
        error_message = str(e)
        
        # 1. Check for Encoding Issues (Sering terjadi di CSV/TXT)
        if "codec" in error_message or "decode" in error_message:
            st.error(f"🔤 **Failed to Read File: Encoding Error**")
            st.warning(
                """
                **Explanation:** The file uses a text format that is not standard UTF-8.
                
                **Solution:** Please try saving your CSV/TXT file specifically as **'UTF-8'** encoding and upload again.
                """
            )

        # 2. Check for PDF Corruption or Password
        elif "PDF" in error_message or "EOF" in error_message:
            st.error(f"📄 **Failed to Read File: Corrupt or Protected**")
            st.info("The PDF file appears to be damaged or password-protected. Please allow printing/extraction or try a different file.")

        # 3. Handle Generic Errors
        else:
            st.error(f"⚠️ **Failed to Process File**")
            st.write(f"**Error Details:** `{error_message}`")
        
        # PENTING: Return None biar aplikasi gak lanjut proses file ini
        return None

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
    