"""Streamlit GUI for the AI Book Reader & QA System.

Features:
- Upload PDF/TXT/DOCX and add to the FAISS-backed index
- See a document list with counts
- Ask a question with configurable top-k retrieval
- View best answer and all candidate contexts with sources and scores
"""

import os
from collections import Counter

import streamlit as st

from document_loader import DocumentLoader
from vector_store import VectorStore
from qa_system import QASystem
import nltk

# Ensure sentence tokenizer is available; fallback to naive splitting
nltk.download('punkt', quiet=True)
try:
    from nltk.tokenize import sent_tokenize
    _HAS_SENT_TOKENIZE = True
except Exception:
    _HAS_SENT_TOKENIZE = False


def chunk_text(text: str, chunk_size: int = 500):
    """Split text into chunks roughly chunk_size words using sentences when possible."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    if _HAS_SENT_TOKENIZE:
        sentences = sent_tokenize(text)
        chunks = []
        current = ''
        for sent in sentences:
            if len((current + ' ' + sent).split()) > chunk_size:
                if current:
                    chunks.append(current.strip())
                current = sent
            else:
                current += ' ' + sent
        if current:
            chunks.append(current.strip())
        return chunks

    # Fallback: simple word-based splitting
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


st.set_page_config(page_title="AI Book Reader", layout="wide")
st.title("📚 AI Book Reader & QA System")

# Initialize store and QA system
vector_store = VectorStore()
qa_system = QASystem(vector_store)


def list_documents():
    """Return a Counter of sources currently in the metadata."""
    sources = [m.get('source', 'unknown') for m in getattr(vector_store, 'metadata', [])]
    return Counter(sources)


with st.sidebar:
    st.header("Add a Book")
    uploaded_file = st.file_uploader("Upload PDF, TXT, or DOCX", type=["pdf", "txt", "docx"])
    if uploaded_file:
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        text = DocumentLoader.load_and_clean(file_path)
        if not text:
            st.error("Could not extract text from the uploaded file.")
        else:
            chunks = chunk_text(text)
            meta = [{'source': uploaded_file.name, 'text': chunk} for chunk in chunks]
            vector_store.add_texts([m['text'] for m in meta], meta)
            st.success(f"Added {len(chunks)} chunks from {uploaded_file.name}")

    st.markdown("---")
    st.header("Index & Documents")
    docs = list_documents()
    if docs:
        for src, cnt in docs.items():
            st.write(f"- {src}: {cnt} chunks")
    else:
        st.write("No documents indexed yet.")

    if st.button("Reset persisted index"):
        # Remove persisted files and reload the app
        try:
            if os.path.exists(vector_store.index_path):
                os.remove(vector_store.index_path)
            if os.path.exists(vector_store.meta_path):
                os.remove(vector_store.meta_path)
            st.success("Deleted persisted index files. The app will reload.")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Could not delete index files: {e}")


st.header("Ask a Question")
col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_input("Enter your question:")
with col2:
    top_k = st.slider("Top-k", min_value=1, max_value=10, value=5)

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            result = qa_system.answer_question(question, top_k=top_k)

        best = result.get('best_answer')
        if not best:
            st.info(result.get('message', 'No answer found or no documents indexed.'))
        else:
            st.subheader("Best Answer")
            st.markdown(f"**Answer:** {best.get('answer','')}  ")
            st.markdown(f"**Score:** {best.get('score', 0):.4f}")
            st.markdown(f"**Source:** {best.get('meta', {}).get('source', 'unknown')}")

            st.markdown("---")
            st.subheader("All Candidate Contexts")
            for i, cand in enumerate(result.get('all_answers', []), start=1):
                with st.expander(f"Candidate {i} — score: {cand.get('score', 0):.4f} — source: {cand.get('meta', {}).get('source','unknown')}"):
                    st.write(cand.get('answer', ''))
                    st.code(cand.get('context', ''))

            st.markdown("---")
            st.caption("Answers are extractive spans from the retrieved contexts. For generative responses, integrate a seq2seq/causal model.")
