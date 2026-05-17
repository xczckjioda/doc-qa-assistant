from fastapi import APIRouter, UploadFile, File
import os

from app.services.file_service import extract_text_by_page
from app.services.chunk_service import chunk_pages
from app.services.embedding_service import EmbeddingService
from app.services.bm25_service import BM25Service
from app.services.vector_store_service import VectorStoreService


router = APIRouter()

UPLOAD_DIR = "data/uploads"

embedding_service = EmbeddingService()
vector_store_service = VectorStoreService()
bm25_service = BM25Service()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    print("Received file:", file.filename)

    pages = extract_text_by_page(file_path)
    chunks = chunk_pages(pages, file.filename)
    for chunk in chunks:
        chunk["file_name"] = file.filename

    print(f"Number of chunks created: {len(chunks)}")
    print([chunk["chunk_id"] for chunk in chunks[:5]])
    print("extract_text_by_page:", file_path)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.embed_texts(texts)

    replaced_existing_file = vector_store_service.file_exists(file.filename)
    if replaced_existing_file:
        vector_store_service.delete_by_file_name(file.filename)

    vector_store_service.add_chunks(chunks, embeddings)
    bm25_service.rebuild_index()

    return {
        "filename": file.filename,
        "message": "Upload successful",
        "mode": "replace" if replaced_existing_file else "append",
        "pages_count": len(pages),
        "chunks_count": len(chunks),
        "preview": chunks[:3]
    }
