# AI Book Reader & Question Answering System

This repository contains a simple AI-powered Book Reader and Question Answering system in Python.

## Contents

- `document_loader.py` – Extracts and cleans text from PDF, TXT, and DOCX files.
- `vector_store.py` – Builds embeddings with SentenceTransformers and stores them in a FAISS index.
- `qa_system.py` – Retrieves relevant text chunks and answers questions using a HuggingFace QA model.
- `main.py` – CLI entry point to add books and ask questions.
- `app.py` – Optional Streamlit web interface.
- `requirements.txt` – Python dependencies.

## Quick setup (Windows / PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. (Optional) If `faiss-cpu` installation fails on Windows, consider using a conda environment:

```powershell
conda create -n ai-reader python=3.10 -y; conda activate ai-reader
conda install -c conda-forge faiss-cpu -y
pip install -r requirements.txt --no-deps
```

4. Add a book (PDF/TXT/DOCX):

```powershell
python main.py --add "C:\\path\\to\\book.pdf"
```

5. Ask a question:

```powershell
python main.py --ask "Who is the protagonist of the book?" --top_k 5
```

6. Run the Streamlit UI:

```powershell
streamlit run app.py
```

## Notes & caveats

- Models and embeddings are downloaded from the internet the first time they are used.
- Large books will consume disk and memory depending on chunking and model sizes.
- On Windows, `faiss` can be tricky to install via pip. Prefer `conda` if you hit errors.
- The project is intentionally minimal. Consider adding persistent metadata storage, more robust chunking (token-based), and batching for large-scale ingestion.

## Next steps / enhancements

- Improve chunking using `tiktoken` or token-based splitting.
- Add unit tests for loader and vector store.
- Add a small web UI with streaming answers and citation highlighting.
