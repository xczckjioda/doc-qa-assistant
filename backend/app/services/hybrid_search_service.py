from typing import List, Optional

from app.services.bm25_service import BM25Service
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService


class HybridSearchService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()
        self.bm25_service = BM25Service()

    def search(
        self,
        query: str,
        top_k: int = 20,
        file_name: Optional[str] = None,
    ) -> List[str]:
        query_embedding = self.embedding_service.embed_query(query)
        vector_results = self.vector_store.search(
            query_embedding,
            top_k=20,
            file_name=file_name,
        )
        vector_chunk_ids = self._extract_vector_ids(vector_results)

        bm25_results = self.bm25_service.search(
            query,
            top_k=20,
            file_name=file_name,
        )
        bm25_chunk_ids = [item["chunk_id"] for item in bm25_results]

        return self._rrf_fuse(
            ranked_lists=[vector_chunk_ids, bm25_chunk_ids],
            top_k=top_k,
            rrf_k=60,
        )

    def _extract_vector_ids(self, raw_results) -> List[str]:
        ids_list = raw_results.get("ids", [])
        if not ids_list or not ids_list[0]:
            return []

        return ids_list[0]

    def _rrf_fuse(
        self,
        ranked_lists: List[List[str]],
        top_k: int,
        rrf_k: int,
    ) -> List[str]:
        scores = {}

        for ranked_list in ranked_lists:
            for rank, chunk_id in enumerate(ranked_list, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

        ranked_chunk_ids = sorted(
            scores,
            key=lambda chunk_id: scores[chunk_id],
            reverse=True,
        )
        return ranked_chunk_ids[:top_k]
