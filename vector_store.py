"""
vector_store.py
Handles embeddings generation and FAISS storage/retrieval.
"""
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


class VectorStore:
    """Manages embeddings and a FAISS index with simple metadata storage.

    - Uses SentenceTransformer to produce embeddings.
    - Persists FAISS index and metadata to disk so the index survives restarts.
    """

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', index_path: str = 'faiss.index', meta_path: str = 'meta.npy'):
        self.model = SentenceTransformer(embedding_model)
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.metadata: List[dict] = []
        self._load_index()

    def _load_index(self):
        """Load an existing FAISS index and metadata if present, otherwise create a new one."""
        dim = self.model.get_sentence_embedding_dimension()
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                self.metadata = np.load(self.meta_path, allow_pickle=True).tolist()
            except Exception:
                # If reading fails, fall back to an empty index
                self.index = faiss.IndexFlatL2(dim)
                self.metadata = []
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []

    def save(self):
        """Persist the FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        np.save(self.meta_path, np.array(self.metadata, dtype=object))

    def add_texts(self, texts: List[str], meta: List[dict]):
        """Encode a list of texts and add their embeddings along with metadata.

        texts and meta must be the same length. Embeddings are added to the FAISS
        index and metadata appended in the same order.
        """
        if not texts:
            return
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # Ensure dtype is float32 for FAISS
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.metadata.extend(meta)
        self.save()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, dict, float]]:
        """Search the vector index for the most similar chunks to the query.

        Returns a list of tuples: (text, metadata, distance)
        """
        if self.index.ntotal == 0:
            return []
        query_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(np.asarray(query_emb, dtype=np.float32), top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.metadata):
                results.append((self.metadata[idx]['text'], self.metadata[idx], float(dist)))
        return results
