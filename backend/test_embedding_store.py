from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService


def main():
    chunks = [
        {
            "chunk_id": "test_1",
            "file_name": "demo.pdf",
            "page": 1,
            "text": "克苏鲁神话是洛夫克拉夫特创造的恐怖宇宙观。"
        },
        {
            "chunk_id": "test_2",
            "file_name": "demo.pdf",
            "page": 2,
            "text": "调查员可以通过技能检定获取线索。"
        },
        {
            "chunk_id": "test_3",
            "file_name": "demo.pdf",
            "page": 3,
            "text": "理智值会因为遭遇超自然现象而下降。"
        }
    ]

    embedding_service = EmbeddingService()
    vector_store = VectorStoreService()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.embed_texts(texts)

    print("chunks 数量:", len(chunks))
    print("embedding 维度:", len(embeddings[0]))

    vector_store.add_chunks(chunks, embeddings)
    print("已写入向量库")

    query = "理智值下降是什么意思？"
    query_embedding = embedding_service.embed_query(query)

    results = vector_store.search(query_embedding, top_k=2)
    print(results)


if __name__ == "__main__":
    main()