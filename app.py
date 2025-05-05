import streamlit as st
import os
import time
import pandas as pd
from document_loader import load_and_split_documents, load_constitution
from vector_store import create_vector_store, load_vector_store
from llm_service import create_qa_chain
from chat_history import save_chat_history, load_chat_history

# Page configuration
st.set_page_config(
    page_title="Kazakhstan Constitution Assistant",
    page_icon="🇰🇿",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "constitution_loaded" not in st.session_state:
    st.session_state.constitution_loaded = False

# App title and description
st.title("Kazakhstan Constitution AI Assistant")
st.markdown("""
This AI assistant can answer questions about the Constitution of the Republic of Kazakhstan 
and any additional documents you upload. The assistant uses vector search technology to find 
relevant information and provide accurate responses.
""")

# Sidebar for app controls
with st.sidebar:
    st.header("Settings")

    # Load Constitution button
    if not st.session_state.constitution_loaded:
        if st.button("📜 Load Constitution of Kazakhstan"):
            with st.spinner("Downloading and processing the Constitution..."):
                try:
                    constitution_docs = load_constitution()
                    if not st.session_state.vectorstore:
                        st.session_state.vectorstore = create_vector_store(constitution_docs)
                    else:
                        st.session_state.vectorstore.add_documents(constitution_docs)
                    
                    st.session_state.qa_chain = create_qa_chain(st.session_state.vectorstore)
                    st.session_state.constitution_loaded = True
                    st.success("✅ Constitution loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading Constitution: {e}")
    else:
        st.success("✅ Constitution is loaded")

    # File uploader
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "📎 Upload PDF/DOCX/TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🔍 Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join("temp_uploads", uploaded_file.name)
                        os.makedirs("temp_uploads", exist_ok=True)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.read())
                        file_paths.append(file_path)

                    documents = load_and_split_documents(file_paths)
                    st.success(f"✅ Loaded {len(documents)} document chunks")

                    if not st.session_state.vectorstore:
                        st.session_state.vectorstore = create_vector_store(documents)
                    else:
                        st.session_state.vectorstore.add_documents(documents)

                    st.session_state.qa_chain = create_qa_chain(st.session_state.vectorstore)
                    st.success("📦 Documents indexed successfully!")

                except Exception as e:
                    st.error(f"❌ Error processing documents: {e}")

    # View chat history
    st.header("Chat History")
    if st.button("View All Chat History"):
        st.session_state.show_history = True

# Main chat interface
st.header("Chat with the Constitution Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the Constitution or uploaded documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.qa_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain.invoke({"question": prompt})
                    answer = response['result']
                    st.markdown(answer)

                    if "source_documents" in response and response["source_documents"]:
                        with st.expander("📚 Sources"):
                            for i, doc in enumerate(response.get("source_documents", [])):
                                st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                                st.markdown(f"**Content:** {doc.page_content[:200]}...")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    save_chat_history(prompt, answer, timestamp)
                    st.session_state.chat_history = load_chat_history()
                except Exception as e:
                    error_msg = f"❌ Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        with st.chat_message("assistant"):
            msg = "Please load the Constitution or upload documents first."
            st.warning(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})

# Show chat history if requested
if st.session_state.get("show_history", False):
    st.header("Chat History")
    if st.session_state.chat_history.empty:
        st.info("No chat history available.")
    else:
        st.dataframe(st.session_state.chat_history, use_container_width=True, hide_index=True)
        csv = st.session_state.chat_history.to_csv(index=False)
        st.download_button(
            "Download Chat History",
            csv,
            "chat_history.csv",
            "text/csv",
            key='download-csv'
        )
    if st.button("Hide History"):
        st.session_state.show_history = False
        st.experimental_rerun()
