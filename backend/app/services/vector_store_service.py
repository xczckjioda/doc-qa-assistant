import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional


CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "documents"


class VectorStoreService:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "file_name": chunk["file_name"],
                "page": chunk["page"],
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        
        print(f"Number of chunks written to Chroma: {len(ids)}")
        print(f"Collection count after write: {self.collection.count()}")

    def file_exists(self, file_name: str) -> bool:
        results = self.collection.get(
            where={"file_name": file_name},
            limit=1,
        )
        return bool(results.get("ids"))

    def delete_by_file_name(self, file_name: str):
        self.collection.delete(
            where={"file_name": file_name},
        )
        print(f"Deleted existing chunks for file: {file_name}")

    def get_all_chunks(self):
        return self.collection.get(
            include=["documents", "metadatas"],
        )

    def list_file_names(self) -> List[str]:
        chunks = self.get_all_chunks()
        metadatas = chunks.get("metadatas", []) or []
        file_names = {
            metadata.get("file_name")
            for metadata in metadatas
            if metadata.get("file_name")
        }
        return sorted(file_names)

    def get_chunks_by_ids(self, chunk_ids: List[str]):
        if not chunk_ids:
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
            }

        return self.collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        file_name: Optional[str] = None,
    ):
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if file_name:
            query_kwargs["where"] = {"file_name": file_name}

        results = self.collection.query(**query_kwargs)
        return results
    
    def reset_collection(self):
        try:
            self.client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )
