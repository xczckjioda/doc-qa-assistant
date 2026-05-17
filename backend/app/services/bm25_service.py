import re
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from app.services.vector_store_service import VectorStoreService


class BM25Service:
    _bm25: Optional[BM25Okapi] = None
    _chunk_ids: List[str] = []
    _documents: List[str] = []
    _metadatas: List[Dict] = []

    def __init__(self):
        self.vector_store = VectorStoreService()

    def rebuild_index(self):
        chunks = self.vector_store.get_all_chunks()

        BM25Service._chunk_ids = chunks.get("ids", []) or []
        BM25Service._documents = chunks.get("documents", []) or []
        BM25Service._metadatas = chunks.get("metadatas", []) or []

        tokenized_documents = [
            self._tokenize(document)
            for document in BM25Service._documents
        ]

        BM25Service._bm25 = (
            BM25Okapi(tokenized_documents)
            if tokenized_documents
            else None
        )

        print(f"BM25 index rebuilt with {len(BM25Service._chunk_ids)} chunks")

    def search(
        self,
        query: str,
        top_k: int = 20,
        file_name: Optional[str] = None,
    ) -> List[Dict]:
        if BM25Service._bm25 is None:
            self.rebuild_index()

        if BM25Service._bm25 is None:
            return []

        query_tokens = self._tokenize(query)
        scores = BM25Service._bm25.get_scores(query_tokens)

        candidates = []
        for index, score in enumerate(scores):
            metadata = (
                BM25Service._metadatas[index]
                if index < len(BM25Service._metadatas)
                else {}
            )
            if file_name and metadata.get("file_name") != file_name:
                continue

            candidates.append(
                {
                    "chunk_id": BM25Service._chunk_ids[index],
                    "score": float(score),
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())
