"""
qa_system.py
Retrieves relevant context and generates answers using an LLM.
"""
from typing import List, Optional
from transformers import pipeline
from vector_store import VectorStore


class QASystem:
    """Retrieve relevant context chunks and answer user questions.

    Uses a QA pipeline (extractive) over the retrieved contexts. If no
    extractive model is provided, this class could be extended to call a
    generative model instead.
    """

    def __init__(self, vector_store: VectorStore, llm_model: str = 'deepset/roberta-base-squad2'):
        self.vector_store = vector_store
        # Pipeline for extractive question-answering (question + context -> span answer)
        self.qa_pipeline = pipeline('question-answering', model=llm_model, tokenizer=llm_model)

    def answer_question(self, question: str, top_k: int = 5) -> dict:
        """Return the best answer and supporting candidates for a question.

        Steps:
        1. Retrieve top_k relevant chunks from vector store.
        2. Run an extractive QA model over each chunk.
        3. Aggregate and return the best-scoring answer and all candidates.
        """
        results = self.vector_store.search(question, top_k=top_k)
        if not results:
            return {'best_answer': None, 'all_answers': [], 'message': 'No documents in the index.'}

        candidates = []
        for context, meta, _ in results:
            qa_input = {'question': question, 'context': context}
            try:
                output = self.qa_pipeline(qa_input)
                candidates.append({
                    'answer': output.get('answer', ''),
                    'score': float(output.get('score', 0)),
                    'context': context,
                    'meta': meta
                })
            except Exception as e:
                candidates.append({'answer': '', 'score': 0.0, 'context': context, 'meta': meta, 'error': str(e)})

        best = max(candidates, key=lambda x: x['score']) if candidates else None
        return {'best_answer': best, 'all_answers': candidates}
