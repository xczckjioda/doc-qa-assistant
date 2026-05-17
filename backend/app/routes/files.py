from fastapi import APIRouter, HTTPException

from app.services.vector_store_service import VectorStoreService


router = APIRouter()
vector_store_service = VectorStoreService()


@router.get("/files")
def list_files():
    try:
        return {"files": vector_store_service.list_file_names()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List files failed: {str(e)}")
