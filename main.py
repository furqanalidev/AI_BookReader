"""
main.py
Entry point for CLI interface.
"""
import argparse
from document_loader import DocumentLoader
from vector_store import VectorStore
from qa_system import QASystem
import nltk
import os

# Try to download punkt tokenizer for sentence splitting. If unavailable,
# falling back to a simple whitespace-based splitter in chunk_text.
nltk.download('punkt', quiet=True)
try:
    from nltk.tokenize import sent_tokenize
    _HAS_SENT_TOKENIZE = True
except Exception:
    _HAS_SENT_TOKENIZE = False


def chunk_text(text: str, chunk_size: int = 500):
    """Split text into chunks roughly chunk_size tokens (words) using sentences.

    Falls back to naive splitting if sentence tokenizer is missing.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    # Prefer sentence-aware splitting for nicer chunks
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
    # Fallback: split by word count
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks


def add_book(file_path: str, vector_store: VectorStore):
    print(f"Loading and processing: {file_path}")
    text = DocumentLoader.load_and_clean(file_path)
    if not text:
        print("No text extracted from the document.")
        return
    chunks = chunk_text(text)
    meta = [{'source': os.path.basename(file_path), 'text': chunk} for chunk in chunks]
    vector_store.add_texts([m['text'] for m in meta], meta)
    print(f"Added {len(chunks)} chunks to vector store.")


def main():
    parser = argparse.ArgumentParser(description="AI Book Reader & QA System")
    parser.add_argument('--add', type=str, help='Path to book file to add (PDF, TXT, DOCX)')
    parser.add_argument('--ask', type=str, help='Question to ask the system')
    parser.add_argument('--top_k', type=int, default=5, help='Number of context chunks to retrieve')
    args = parser.parse_args()

    vector_store = VectorStore()
    qa_system = QASystem(vector_store)

    if args.add:
        add_book(args.add, vector_store)
        return

    if args.ask:
        result = qa_system.answer_question(args.ask, top_k=args.top_k)
        best = result.get('best_answer')
        if not best:
            print(result.get('message', 'No answer found.'))
            return
        print(f"\nAnswer: {best.get('answer', '')}\nScore: {best.get('score', 0):.4f}")
        print(f"\nReference context (source: {best.get('meta', {}).get('source', 'unknown')}):\n{best.get('context', '')}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
