from app.services.hybrid_search_service import HybridSearchService
from app.services.vector_store_service import VectorStoreService
from app.services.llm_service import LLMService
from app.services.rerank_service import RerankService
import re

class RetrievalService:
    def __init__(self):
        self.hybrid_search_service = HybridSearchService()
        self.vector_store = VectorStoreService()
        self.llm_service = LLMService()
        self.rerank_service = RerankService()

    def retrieve(self, query: str, top_k: int = 5, file_name: str | None = None):
        rewritten_query = self._rewrite_query(query)
        original_chunk_ids = self.hybrid_search_service.search(
            query,
            top_k=20,
            file_name=file_name,
        )
        rewritten_chunk_ids = self.hybrid_search_service.search(
            rewritten_query,
            top_k=20,
            file_name=file_name,
        )
        chunk_ids = self._rrf_fuse(
            ranked_lists=[original_chunk_ids, rewritten_chunk_ids],
            top_k=20,
            rrf_k=60,
        )
        raw_results = self.vector_store.get_chunks_by_ids(chunk_ids)

        ids = raw_results.get("ids", []) or []
        documents = raw_results.get("documents", []) or []
        metadatas = raw_results.get("metadatas", []) or []

        if not ids:
            return {
                "answer": "No relevant information found.",
                "rewritten_query": rewritten_query,
                "sources": [],
                "results": []
            }

        chunks_by_id = {}
        for i, chunk_id in enumerate(ids):
            chunks_by_id[chunk_id] = {
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
            }

        results = []
        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if not chunk:
                continue

            metadata = chunk["metadata"]
            document = chunk["document"]

            text = document.strip()

            if is_noisy_chunk(text):
                continue

            results.append({
                "chunk_id": chunk_id,
                "text": document,
                "metadata": metadata,
                "file_name": metadata.get("file_name"),
                "page": metadata.get("page"),
                "distance": None,
            })

        

        if not results:
            return {
                "answer": "The document does not contain enough relevant information to answer this question.",
                "rewritten_query": rewritten_query,
                "sources": [],
                "results": []
            }
        
        reranked_results = self.rerank_service.rerank(
            query=query,
            chunks=results,
            top_k=5,
        )

        llm_output = self.llm_service.generate_answer(query=query, results=reranked_results)

        return {
            "answer": llm_output["answer"],
            "rewritten_query": rewritten_query,
            "sources": llm_output["sources"],
            "results": reranked_results
        }

    def _rewrite_query(self, query: str) -> str:
        try:
            rewritten_query = self.llm_service.rewrite_query(query)
            return rewritten_query.strip() or query
        except Exception as e:
            print(f"Query rewrite failed: {e}")
            return query

    def _rrf_fuse(self, ranked_lists: list[list[str]], top_k: int, rrf_k: int) -> list[str]:
        scores = {}

        for ranked_list in ranked_lists:
            for rank, chunk_id in enumerate(ranked_list, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

        return sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:top_k]
    
def is_noisy_chunk(text: str) -> bool:
    text = text.strip()

    if not text:
        return True

    if len(text) < 30:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True

    lowered = text.lower()

    # 1. 几乎全是链接
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return True

    # 2. URL 占比过高
    url_count = len(re.findall(r"https?://\S+|www\.\S+", text))
    if url_count > 0 and url_count >= max(1, len(lines) // 2):
        return True

    # 3. 字母数字太少，信息密度低
    alnum_count = sum(ch.isalnum() for ch in text)
    if alnum_count < 20:
        return True

    # 4. 单行短标题 / 页眉页脚风格文本
    if len(lines) <= 2 and len(text) < 50:
        return True

    return False
