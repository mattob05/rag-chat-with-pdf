import streamlit as st
import os
from src import VectorEngine, Reranker, LLMEngine, RAGChain, load_and_chunk_pdf

st.set_page_config(page_title="RAG Knowledge Manager", layout="wide")
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
    
@st.cache_resource(show_spinner="Loading models... Please wait.")
def init_rag_system():
    ve = VectorEngine()
    re = Reranker()
    le = LLMEngine()
    return RAGChain(ve, re, le)

rag = init_rag_system()

with st.sidebar:
    st.title("📂 Knowledge Base")
    
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf", 
        key=f"pdf_uploader_{st.session_state.uploader_key}"
    )
    if uploaded_file and st.button("Add to Database"):
        with st.spinner("Processing..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                chunks = load_and_chunk_pdf(temp_path)
                
                for chunk in chunks:
                    chunk.metadata["source"] = uploaded_file.name
                
                rag.vector_engine.add_documents(chunks)
                st.session_state.uploader_key += 1
                st.success(f"Successfully added {uploaded_file.name}!")
                st.rerun()
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    st.divider()

    st.subheader("Manage Files")
    files = rag.vector_engine.get_all_files()
    
    if not files:
        st.info("No documents in database.")
    else:
        for f in files:
            col1, col2 = st.columns([0.8, 0.2], vertical_alignment="center")
            
            col1.markdown(f"📄 **{f}**")
            
            if col2.button("🗑️", key=f, help=f"Delete {f}"):
                with st.spinner(f"Deleting {f}..."):
                    rag.vector_engine.delete_file_by_name(f)
                st.rerun()

st.title("🤖 Chat with your Documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag.get_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})