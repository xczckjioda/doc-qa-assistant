from typing import Dict, List

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankService:
    def __init__(self):
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        if not chunks:
            return []

        pairs = [
            [query, chunk.get("text", "")]
            for chunk in chunks
        ]
        scores = self.model.predict(pairs)

        reranked_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk_with_score = dict(chunk)
            chunk_with_score["rerank_score"] = float(score)
            reranked_chunks.append(chunk_with_score)

        reranked_chunks.sort(
            key=lambda chunk: chunk["rerank_score"],
            reverse=True,
        )
        return reranked_chunks[:top_k]
