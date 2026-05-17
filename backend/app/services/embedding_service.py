from sentence_transformers import SentenceTransformer
from typing import List


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode([query], convert_to_numpy=True)[0]
        return embedding.tolist()